class Response:
    def __init__(self, client, model_name, system_prompt, temperature, top_p, chat_history):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.chat_history = chat_history

    def get_response(self, user_input):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *[
                        {"role": "user", "content": m["content"]}
                        for m in self.chat_history if m["role"] == "user"
                    ],
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=8192,
                top_p=self.top_p
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return None
