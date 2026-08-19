import logging
from typing import Any

import litellm
from litellm import acompletion, completion

from core.base.abstractions import GenerationConfig
from core.base.providers.llm import CompletionConfig, CompletionProvider
from core.billing import BillingUsageRecorder

logger = logging.getLogger()


class LiteLLMCompletionProvider(CompletionProvider):
    def __init__(
        self,
        config: CompletionConfig,
        *args,
        billing_usage_recorder: BillingUsageRecorder | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config)
        litellm.modify_params = True
        self.acompletion = acompletion
        self.completion = completion
        self.billing_usage_recorder = billing_usage_recorder

        # if config.provider != "litellm":
        #     logger.error(f"Invalid provider: {config.provider}")
        #     raise ValueError(
        #         "LiteLLMCompletionProvider must be initialized with config with `litellm` provider."
        #     )

    def _get_base_args(
        self, generation_config: GenerationConfig
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": generation_config.model,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "stream": generation_config.stream,
            "max_tokens": generation_config.max_tokens_to_sample,
            "api_base": generation_config.api_base,
        }

        # Fix the type errors by properly typing these assignments
        if generation_config.functions is not None:
            args["functions"] = generation_config.functions
        if generation_config.tools is not None:
            args["tools"] = generation_config.tools
        if generation_config.response_format is not None:
            args["response_format"] = generation_config.response_format

        return args

    async def _execute_task(self, task: dict[str, Any]):
        messages = task["messages"]
        generation_config = task["generation_config"]
        kwargs = task["kwargs"]

        args = self._get_base_args(generation_config)
        args["messages"] = messages
        args = {**args, **kwargs}

        billing_event_id = None
        if self.billing_usage_recorder is not None:
            billing_event_id = await self.billing_usage_recorder.start_call(
                operation="completion",
                model=args["model"],
            )
            args = self.billing_usage_recorder.add_litellm_metadata(
                kwargs=args,
                event_id=billing_event_id,
            )

        logger.debug(
            f"Executing LiteLLM task with generation_config={generation_config}"
        )

        try:
            response = await self.acompletion(**args)
        except Exception as error:
            if self.billing_usage_recorder is not None:
                await self.billing_usage_recorder.fail_call(
                    billing_event_id, error
                )
            raise

        if self.billing_usage_recorder is not None:
            await self.billing_usage_recorder.complete_call(
                billing_event_id, response
            )
        return response

    def _execute_task_sync(self, task: dict[str, Any]):
        messages = task["messages"]
        generation_config = task["generation_config"]
        kwargs = task["kwargs"]

        args = self._get_base_args(generation_config)
        args["messages"] = messages
        args = {**args, **kwargs}

        logger.debug(
            f"Executing LiteLLM task with generation_config={generation_config}"
        )

        try:
            return self.completion(**args)
        except Exception as e:
            logger.error(f"Sync LiteLLM task execution failed: {str(e)}")
            raise
