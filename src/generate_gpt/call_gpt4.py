import backoff
import openai
from openai import AzureOpenAI, AsyncAzureOpenAI
import itertools
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout


error_types = (RequestException, HTTPError, ConnectionError, Timeout, ValueError)

# client_turbo = AzureOpenAI(
#     api_version="2023-07-01-preview",
#     api_key="568cbb18b12c46bfbde5904fc4b99a09",
#     azure_endpoint="https://gpt4v-jb.openai.azure.com/",
# )

from openai import AsyncAzureOpenAI
import backoff

client_turbo = AsyncAzureOpenAI(
    api_key="e849b8c4c4a04d3d817aa67d66189251",
    api_version="2024-02-01",
    azure_endpoint="https://jb-turbo-2024-04-09.openai.azure.com/",
)

# Define the error types to catch for backoff
error_types = (Exception,)  # Replace with specific error types if known


@backoff.on_exception(backoff.expo, error_types, max_tries=1)
async def completions_with_backoff_turbo(**kwargs):
    return await client_turbo.chat.completions.create(**kwargs)


async def call_gpt4_turbo(system_prompt, prompt, temperature=0, n=1):
    response = await completions_with_backoff_turbo(
        model="jb-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        n=n,
    )
    cost = response.usage.completion_tokens * (
        0.03 / 1000
    ) + response.usage.prompt_tokens * (0.01 / 1000)
    completion = response.choices[0].message.content
    return cost, completion


client_4v = AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key="568cbb18b12c46bfbde5904fc4b99a09",
    azure_endpoint="https://gpt4v-jb.openai.azure.com/",
)


@backoff.on_exception(backoff.expo, error_types)
def completions_with_backoff_4v(**kwargs):
    return client_4v.chat.completions.create(**kwargs)


def call_gpt4v(system_prompt, prompt, temperature=0, n=1, max_tokens=256):
    response = completions_with_backoff_4v(
        model="GPT4V",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        n=n,
        max_tokens=max_tokens,
    )
    cost = response.usage.completion_tokens * (
        0.03 / 1000
    ) + response.usage.prompt_tokens * (0.01 / 1000)
    completion = response.choices[0].message.content
    return cost, completion
