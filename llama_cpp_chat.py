import asyncio
import collections
import time

from llama_cpp import Llama

class Llama_Chat(object):
    @classmethod
    async def create(cls, bot_prompt:str, llm_model_path:str, n_ctx:int=1024, n_gpu_layers=0, llm_config:dict={"frequency_penalty":1.0, "presence_penalty":1.0, "top_k": 30, 
        "top_p": 0.95, "typical_p":0.5, "min_p": 0.1, "temperature": 0.8, "repeat_penalty": 1.1, "max_tokens": 128, "stop": []}, max_message_history:int=30, 
        reply_ratio:float=0.75, logger=None) -> None:
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
        chat_init = {'bot_prompt':self.bot_prompt, 'chat_log':collections.deque(maxlen=self.max_message_history), 'impersonate':None, 'summaries': collections.deque(maxlen=int(self.max_message_history*0.1) or 1)}

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
        Manages the llm object to save memory when not in use.
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
            response = self.llm(prompt, **self.llm_config, seed=int(time.time()))
            return response['choices'][0]['text'].strip()

        async def make_prompt(channel_info:dict, message:dict, reply_chain:list=None) -> str:
            """
            Combines the channel chat history and recent reply chain with the bot's prompt.  
            """
            chat_log = list(channel_info['chat_log'])
            summaries = channel_info['summaries']

            bot_name = channel_info['bot_name'] if not channel_info['impersonate'] else channel_info['impersonate']

            # If the message is for a command, use the right prompt for it, otherwise use the default chatbot prompt.
            if message['action'] == None: 
                # summarise here
                if len(chat_log) >= int(self.max_message_history*0.8):
                    num_to_summarize = int(self.max_message_history * 0.8 / 2)
                    messages_to_summarise = chat_log[:num_to_summarize]
                    
                    # Prepare messages for summarization
                    messages_to_summarise.insert(0, f"{bot_name}: {channel_info['bot_prompt']}")
                    messages_to_summarise.insert(1, "    - Reply in a single message.")
                    messages_to_summarise.insert(2, "    - Mind the chat history when writing a response to encourage open-endedness and continuity.")
                    messages_to_summarise.insert(3, "    - Fight apathy and boredom with thought out creative charactisation.")
                    messages_to_summarise.insert(4, f"--- {channel_info['channel_name']} Chat History ---")

                    summary_prompt = (
                        f"{'\n'.join(messages_to_summarise)}\n"
                        "--- END Chat History ---\n"
                        f"As {bot_name}, how do you feel about the conversation so far? What themes or emotions do you think are emerging?\n"
                        f"{bot_name}:"
                    )

                    summary = await asyncio.to_thread(generate_text, summary_prompt)
                    #    print(summary)
                    summary = f"{bot_name}: {summary}"
                    summaries.append(summary)

                    del chat_log[:num_to_summarize]
                    for item in range(num_to_summarize):
                        del channel_info['chat_log'][0]

                if reply_chain:
                    chat_log = [msg for msg in chat_log if msg not in reply_chain]
                    # shrink the chat_log in favour of the reply_chain while keeping within the max_message_history
                    # works because deques will keep the length within the max_message_history
                    if len(reply_chain) + len(chat_log) > self.max_message_history:
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

                    chat_log.extend(reply_chain)

                chat_log.insert(0, f"{bot_name}: {channel_info['bot_prompt']}")
                chat_log.insert(1, "    - Mind the chat history when writing a response to encourage open-endedness and continuity.")
                chat_log.insert(2, "    - Fight apathy and boredom with thought out creative charactisation.")
                chat_log.insert(3, f"--- {channel_info['channel_name']} Chat History ---")
                if summaries:
                    chat_log.insert(4, '\n'.join(summaries))
                chat_log.append("--- END Chat History ---")
                chat_log.append(f"{message['username']}: {message['content']}")
                chat_log.append(f"{bot_name}:")
            else:
                if message['action'] == 'impersonate':
                    chat_log.clear()
                    chat_log.append(f"Given the name {message['content']}, write a short but detailed introduction that includes striking elements for character in a descriptive manner.")
                    chat_log.append(f"--- System Character Creator ---")
                    chat_log.append(f"{message['content']}:")
                else:
                    raise ValueError(f"{message['action']} is not a valid action.")

            prompt = '\n'.join(chat_log)
            return prompt

        async def process_message(channel_info:dict, future:asyncio.Future, message:dict) -> str:
            """
            Prepare and send a response for the user message using a channel specific prompt.
            """
            prompt = await make_prompt(channel_info, message, message['reply_list'])
            response = await asyncio.to_thread(generate_text, prompt)
            
            bot_responses = []
            for line in response.split('\n'):
                if line.startswith(channel_info['bot_name']) or response.index(line) == 0:
                    line = line.replace(f'{channel_info['bot_name']}: ', '').strip()
                    bot_responses.append(line)
                    if channel_info['impersonate'] and not message['action']:
                        line = f"(As {channel_info['impersonate']}):"+" "+line
                else:
                    break

            # Save the messages together for better consistency.
            if message['action'] == None:
                channel_info['chat_log'].append(f"{message['username']}: {message['content']}")
                channel_info['chat_log'].extend([f'{channel_info['bot_name']}: ' + response.strip() for response in bot_responses])
            future.set_result(bot_responses) 

        while True:
            ### Worker Loop ###
            try:
                channel_info, future, message = await self.text_queue.get()
                await process_message(channel_info, future, message)
            except Exception as exception:
                if self.logger:
                    self.logger.error(f"Llama_Chat encountered an error while replying:\n{exception}")
                future.set_exception(exception)
