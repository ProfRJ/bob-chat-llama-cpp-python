import asyncio
import collections
import time

from llama_cpp import Llama

class Llama_Chat(object):
    @classmethod
    async def create(cls, bot_prompt:str, llm_model_path:str, n_ctx:int=1024, n_gpu_layers=0, llm_config:dict={"top_k": -1, "top_p": 0.95, "typical_p":1.0, "min_p": 0, 
    "temperature": 1.3, "repetition_penalty": 1.2, "max_tokens": 128, "stop": ["\n"]}, max_message_history:int=100, reply_ratio:float=0.75, logger=None) -> None:
        """
        Creates the worker class, initialising it with the given config.

        bot_prompt - str containing a description about the bot.
        llm_model_path - str or pathlike object that points to the location of the llm.
        context_length - int setting how many tokens of context can be used for each generation. (default 1024)
        gpu_layers - int setting how much of the model can be put onto the gpu requires cuda. (default 0)
        llm_config - dict containing the settings used for each request to the llm. (see https://docs.vllm.ai/en/latest/dev/sampling_params.html)
        max_message_history - int setting the message cap for each channel. (default 100)
        reply_ratio - float setting how much the channel chatlog will shrink to fit in the max_message_history if given a long enough reply_list. (default 0.75)
        logger - logging object (default None)
        """
        self = Llama_Chat()
        self.bot_prompt = bot_prompt
        self.break_idle = asyncio.Event()
        self.channels = {}
        self.n_ctx = n_ctx
        self.llm_model_path = llm_model_path
        self.idle = True
        self.idle_manager = asyncio.create_task(self._idle_manager())
        self.llm_config = llm_config
        self.logger = logger
        self.max_message_history = max_message_history
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.reply_ratio = reply_ratio
        self.text_queue = asyncio.Queue(maxsize=0)
        self.text_worker = asyncio.create_task(self._text_worker())
        return self
        
    async def __call__(self, content:str, action:str=None, channel_info:dict=None, reply_list:[str]=None, username:str='user'):
        """
        Formats the incoming message to send to the llm, which then returns the llm's response asynchronously.
        
        content - str of the given message.
        action - How should the prompt be formatted? [None = chatbot, 'impersonate' = character description] (default None)
        channel_info - dict of channel_info supplied from self.get_channel_info. No value will use a single channel for all requests. (default None)
        reply_list - list of previous replies to merge into the prompt. (default None)
        username - str of the message's author. (default 'user')
        """
        self.last_message_time = time.time()
        if self.idle:
            self.idle = False
            self.break_idle.set()

        if not channel_info:
            channel_info = self.get_channel_info()
        if reply_list:
            # dodgy code that ensures the bot replies have the right identity in the reply_chain 
            for index, message in enumerate(reply_list):
                message = message.split(':')
                author, text = message[0], message[1].strip()
                impersonate = None
                if text.startswith('(As '):
                    impersonate = text[text.find('(As ')+4:text.find('): ')]
                    text = text[text.find('): ')+3:]
                    reply_list[index] = f"{impersonate}: {text}"

        message = {'action':action, 'content':content, 'reply_list':reply_list, 'username':username}

        future = asyncio.Future()
        await self.text_queue.put((channel_info, future, message))
        return await future
        
    def get_channel_info(self, bot_name:str="Bob", channel_id:str='1', channel_name:str=None, server_id:str='system') -> dict:
        """
        Retreives data about the channel of the given message, creating an entry if it does not exist.

        
        bot_name - str of the chatbot's name. (default 'Bob')
        channel_id - str of an identifier to store the channel information under. (default '1')
        channel_name - optional str of the channel's name to be used in the prompt. (default f'{bot_name.replace(" ", "-").lower()}s-place')
        server_id - str of an identifier to store each channel under. (default 'system')
        """
        chat_init = {'bot_prompt':self.bot_prompt, 'chat_log':collections.deque(maxlen=self.max_message_history), 'impersonate':None}

        server = self.channels.setdefault(str(server_id), {})
        channel_info = server.setdefault(str(channel_id), chat_init)
        channel_info['bot_name'] = bot_name
        if not channel_name:
            channel_info['channel_name'] = f'{bot_name.replace(" ", "-").lower()}s-place'
        else:
            channel_info['channel_name'] = channel_name
        return channel_info

    async def _idle_manager(self):
        """
        Manages the llm to save memory when not in use.
        """        
        while True:
            self.break_idle.clear()
            if not self.idle:
                if self.logger:
                    self.logger.info('Waking up the LLM')
                self.llm = Llama(model_path=self.llm_model_path, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx, verbose=False)
                current_time = time.time()
                while not current_time-self.last_message_time > 60:
                    await asyncio.sleep(60)
                    current_time = time.time()
                if self.logger:
                    self.logger.info('Putting the LLM to sleep')
                self.idle = True
                del self.llm
            else:
                await self.break_idle.wait()

    async def _text_worker(self) -> None:
        """
        This asynchronous worker loop is responsible for processing messages one by one from the FIFO text_queue.
        """
        def generate_text(prompt:str) -> str:
            """
            Generate a response based on the prompt.
            """
            response = self.llm(prompt, **self.llm_config)
            return response['choices'][0]['text'].strip()

        async def make_prompt(channel_info:dict, message:dict, reply_chain:list=None) -> str:
            """
            Combines the channel chat history and recent reply chain with the bot's prompt.  
            """
            chat_log = list(channel_info['chat_log'])
            chat_log.insert(0, f"--- {channel_info['channel_name']} Channel History ---")
            if reply_chain:
                chat_log = [msg for msg in chat_log if msg not in reply_chain]

                # shrink the chat_log in favour of the reply_chain while keeping within the max_message_history
                # works because deques will keep the length within the max_message_history
                if len(reply_chain)-1 + len(chat_log) > self.max_message_history:
                    reply_ratio = int(self.reply_ratio*self.max_message_history)
                    reply_ratio_flipped = int(reply_ratio+self.max_message_history-reply_ratio*2)

                    # if the reply_chain is big enough, crop it to fit within the specified ratio
                    if len(reply_chain) > reply_ratio:
                        excess_reply_chain = len(reply_chain)-reply_ratio 
                        del reply_chain[:excess_reply_chain]

                    # crop the chat_log as the reply_chain grows
                    spare_space = reply_ratio-len(reply_chain)
                    if len(chat_log) > reply_ratio_flipped and spare_space >= 0:
                        del chat_log[:reply_ratio-spare_space]
                    if len(chat_log) > reply_ratio_flipped and spare_space < 0:
                        del chat_log[:reply_ratio]
                                 
                reply_chain.insert(0, "--- Recent Reply Chain History ---")
                chat_log.extend(reply_chain)

            bot_prompt = f"A few things to keep in mind about {channel_info['bot_name']}: {channel_info['bot_prompt']}"
            
            # If the message is for a command, use the right prompt for it, otherwise use the default chatbot prompt.
            if message['action'] == None: 
                chat_log.append(f"--- System Character Response ---")
                chat_log.append(bot_prompt)
                chat_log.append(f"Given the message from {message['username']}, write a short message in response like {channel_info['bot_name']} would.")
                chat_log.append(f"{message['username']}: {message['content']}")
                chat_log.append(f"{channel_info['bot_name']}:")
            else:
                if message['action'] == 'impersonate':
                    chat_log.append("--- System Character Creator ---")
                    chat_log.append(f"Given the name {message['content']}, write a short characterization that includes striking elements for character in a short but descriptive manner.")
                    chat_log.append(f"{message['content']}:")

            prompt = '\n'.join(chat_log)
            return prompt

        async def process_message(channel_info, future, message) -> str:
            """
            Prepare and send a response for the user message using a channel specific prompt.
            """
            prompt = await make_prompt(channel_info, message, message['reply_list'])
            response = await asyncio.to_thread(generate_text, prompt)
            # Save the message pair together for better consistency.
            if message['action'] == None:
                channel_info['chat_log'].append(f"{message['username']}: {message['content']}")
                channel_info['chat_log'].append(f"{channel_info['bot_name']}: {response}")

            if channel_info['impersonate'] and not message['action']:
                response = f"(As {channel_info['impersonate']}):"+" "+response
            future.set_result(response) 

        while True:
            ### Worker Loop ###
            try:
                channel_info, future, message = await self.text_queue.get()
                await process_message(channel_info, future, message)
            except Exception as exception:
                exception = f"{type(exception).__name__}: {exception}"
                if self.logger:
                    self.logger.error(f"{__class__.__name__} encountered an error while replying:\n{exception}")
                future.set_exception(exception)
