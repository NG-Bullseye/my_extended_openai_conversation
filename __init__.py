"""The OpenAI Conversation integration."""
from __future__ import annotations

import json
import logging
from typing import Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import AuthenticationError, OpenAIError
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
import yaml

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    get_function_executor,
    is_azure,
    validate_authentication,

)
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""
    STATIC_CONVERSATION_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    RELFEKTION_ITERATION_NUMBER = 3

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        base_url = entry.data.get(CONF_BASE_URL)
        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(
                api_key=entry.data[CONF_API_KEY],
                azure_endpoint=base_url,
                api_version=entry.data.get(CONF_API_VERSION),
                organization=entry.data.get(CONF_ORGANIZATION),
            )
        else:
            self.client = AsyncOpenAI(
                api_key=entry.data[CONF_API_KEY],
                base_url=base_url,
                organization=entry.data.get(CONF_ORGANIZATION),
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL
############################################################################################################################################################################################
############################################################################################################################################################################################
    async def async_process(
            self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        # Verwende immer die statische Conversation ID
        conversation_id = self.STATIC_CONVERSATION_ID
        user_input.conversation_id = conversation_id

        exposed_entities = self.get_exposed_entities()

        if conversation_id in self.history:
            messages = self.history[conversation_id]
        else:
            try:
                system_message = self._generate_system_message(
                    exposed_entities, user_input
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [system_message]

        user_message = {"role": "user", "content": user_input.text}
        if self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME):
            user = await self.hass.auth.async_get_user(user_input.context.user_id)
            if user is not None and user.name is not None:
                user_message[ATTR_NAME] = user.name

        messages.append(user_message)

        try:

            filtered_messages = await self.information_extraction_layer(user_input, messages)  # Expecting a list of messages back
            prepared_data = await self.post_query_preparation(user_input, filtered_messages, exposed_entities)
            reflected_data = await self.query_reflection(prepared_data)
                
            query_response = await self.query(
                reflected_data['user_input'], 
                reflected_data['messages'], 
                reflected_data['exposed_entities'], 
                0
            )

            #query_response = await self.query("USERS INPUT:\n "+user_text +"\n GPT RESPONSE: \n"+ response_text +" \n HERE IS YOUR TASK: I want you to improve the Response. If the requirements are not met, i want you to improve it so the user is satisfied. If the user is unspecific, i want you to be creative and execute things you could fit.", messages, exposed_entities, 0)
            user_text=user_input.text
            response_text=query_response.message.content

            # Reflexion und Verbesserung der Antwort
            improved_response = await self.reflect_and_improve_response( user_text, response_text )
               

            query_response.message.content = improved_response


        except OpenAIError as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(query_response.message.model_dump(exclude_none=True))
        self.history[conversation_id] = messages

        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": query_response.response.model_dump(),
                "user_input": user_input,
                "messages": messages,
            },
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(query_response.message.content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )
##############################################################################################
    async def information_extraction_layer(self, user_input: conversation.ConversationInput, messages):
        reflection_prompt = f"Benutzeranfrage: {user_input.text}---\nAufgabe: Bitte extrahiere..."
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": reflection_prompt}],
            max_tokens=150,
            top_p=self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            temperature=self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
        new_content = response.choices[0].message.content
        messages.append({"role": "system", "content": new_content})
        return messages  # Returning a list of messages
##############################################################################################
    async def post_query_preparation(self, user_input: conversation.ConversationInput, messages, exposed_entities):
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        response: ChatCompletion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            top_p=self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            temperature=self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            user=user_input.conversation_id
        )
        choice: Choice = response.choices[0]
        message = choice.message.content if choice.message.content else "No suggestion generated"
        return {"user_input": user_input, "messages": messages, "exposed_entities": exposed_entities, "suggestion": message}
  
##############################################################################################

##############################################################################################
    async def query_reflection(self, prepared_data):
        """Reflect and possibly enrich the prepared query based on user's intent."""
        # Initial access to data from previous steps
        user_input = prepared_data['user_input']
        suggestion = prepared_data['suggestion']
        exposed_entities = prepared_data['exposed_entities']
        
        # First GPT call to assess adequacy of the response
        adequacy_response = await self.is_response_adequate(suggestion, user_input)

        if adequacy_response['adequate'] == 'yes':
            reflected_data = prepared_data  # If adequate, proceed without changes
        else:
            # Second GPT call to improve response based on the feedback
            improvement_prompt = f"User input: {user_input.text}\nInitial suggestion: {suggestion}\nFeedback: {adequacy_response['feedback']}\n---\nTask: Generate an improved suggestion that fulfills the user's intent and addresses the feedback provided."
            improved_suggestion = await self.generate_improved_suggestion(improvement_prompt)

            reflected_data = {
                "user_input": user_input,
                "messages": prepared_data['messages'],
                "exposed_entities": exposed_entities,
                "suggestion": improved_suggestion
            }

        return reflected_data
##############################################################################################

    from pydantic import BaseModel
    from openai import OpenAI

    class AdequacyResponse(BaseModel):
        adequate: str
        feedback: str

    client = OpenAI()

    async def is_response_adequate(self, suggestion, user_input):
        """Use GPT to assess if the suggestion matches the user's intent."""
        evaluation_prompt = f"User input: {user_input.text}\nSuggestion: {suggestion}\n---\nTask: Evaluate the adequacy of the suggestion. Respond with a structured output."

        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": evaluation_prompt,
                },
            ],
            tools=[
                openai.pydantic_function_tool(AdequacyResponse),
            ],
        )

        # The structured output is parsed and returned as a dictionary
        return completion.choices[0].message.tool_calls[0].function.parsed_arguments

##############################################################################################

    async def generate_improved_suggestion(self, prompt):
        """Generate an improved response based on the provided prompt."""
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            top_p=self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            temperature=self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
        return response.choices[0].message.content
##############################################################################################

##############################################################################################
    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        """Process a sentence."""
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        context_threshold = self.entry.options.get(
            CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
        )
        functions = list(map(lambda s: s["spec"], self.get_functions()))
        function_call = "auto"
        if n_requests == self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_call = "none"

        tool_kwargs = {"functions": functions, "function_call": function_call}
        if use_tools:
            tool_kwargs = {
                "tools": [{"type": "function", "function": func} for func in functions],
                "tool_choice": function_call,
            }

        if len(functions) == 0:
            tool_kwargs = {}

        _LOGGER.info("Prompt for %s: %s", model, messages)

        response: ChatCompletion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            user=user_input.conversation_id,
            **tool_kwargs,
        )

        _LOGGER.info("Response %s", response.model_dump(exclude_none=True))

        if response.usage.total_tokens > context_threshold:
            await self.truncate_message_history(messages, exposed_entities, user_input)

        choice: Choice = response.choices[0]
        message = choice.message
     
        # Check if message.content is None and handle it appropriately
        if message.content is None:
            new_text = "default message"  # Use a default message or handle the situation as needed
        else:
            new_text = message.content   

        message.content = new_text
        if choice.finish_reason == "function_call":
            return await self.execute_function_call(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "tool_calls":
            return await self.execute_tool_calls(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "length":
            raise TokenLengthExceededError(response.usage.completion_tokens)

        return OpenAIQueryResponse(response=response, message=message)   
##############################################################################################
    async def reflect_and_improve_response(self, user_query: str, gpt_response: str) -> str:  
        reflection_prompt_1 = f"Benutzeranfrage: {user_query}\nGPT-Antwort: {gpt_response}\n---\nHier ist deine Aufgabe: Schreibe eine Antwort, die ein intelligentes Wesen auf die request den nutzers geben würde. Beachte die ursprüngliche Frage und Antwort vorherirger GPT instanzen. Ich möchte dass die Conversation wie mit Jarvis abläuft. Deine Antwort wird direkt dem Nutzer vorgelesen. Wenn du eine execution function aufrufst gib keine antwort! denn beim ausführen von aktionen soll nicht geredet werden. Außerdem achte darauf, dass die Antworten kurz und informationsdicht sind.'. Just for reference, the previous instance that you get the input from was checking if the requirements are met and improve the response."

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",  # Annahme, dass dies das gewünschte Modell ist
            messages=[
                {"role": "system", "content": reflection_prompt_1}
            ],
            max_tokens=150,
            top_p=self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            temperature=self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        )
        # Gehe davon aus, dass die Antwort verbessert wurde
        return response.choices[0].message.content

############################################################################################################################################################################################
############################################################################################################################################################################################




    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ):
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt = self._async_generate_prompt(raw_prompt, exposed_entities, user_input)
        return {"role": "system", "content": prompt}

    def _async_generate_prompt(
        self,
        raw_prompt: str,
        exposed_entities,
        user_input: conversation.ConversationInput,
    ) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def get_functions(self):
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def truncate_message_history(
        self, messages, exposed_entities, user_input: conversation.ConversationInput
    ):
        """Truncate message history."""
        strategy = self.entry.options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            last_user_message_index = None
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]
                # refresh system prompt when all messages are deleted
                messages[0] = self._generate_system_message(
                    exposed_entities, user_input
                )

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        function_name = message.function_call.name
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ) -> OpenAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(message.function_call.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": message.function_call.name,
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        messages.append(message.model_dump(exclude_none=True))
        for tool in message.tool_calls:
            function_name = tool.function.name
            function = next(
                (s for s in self.get_functions() if s["spec"]["name"] == function_name),
                None,
            )
            if function is not None:
                result = await self.execute_tool_function(
                    user_input,
                    tool,
                    exposed_entities,
                    function,
                )

                messages.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result),
                    }
                )
            else:
                raise FunctionNotFound(function_name)
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_function(
        self,
        user_input: conversation.ConversationInput,
        tool,
        exposed_entities,
        function,
    ) -> OpenAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(tool.function.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(tool.function.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )
        return result


class OpenAIQueryResponse:
    """OpenAI query response value object."""

    def __init__(
        self, response: ChatCompletion, message: ChatCompletionMessage
    ) -> None:
        """Initialize OpenAI query response value object."""
        self.response = response
        self.message = message
