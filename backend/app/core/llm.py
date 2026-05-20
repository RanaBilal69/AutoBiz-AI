import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Explicitly load the .env from the project root, overriding any cached values
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# TEMPORARY DEBUG
print(f"[DEBUG] .env path: {env_path}")
print(f"[DEBUG] .env exists: {env_path.exists()}")
print(f"[DEBUG] API KEY: '{os.getenv('OPENAI_API_KEY', 'NOT FOUND')[:15]}...'")


def get_llm():
    model    = os.getenv("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    api_key  = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.1,
        max_retries=3,
        request_timeout=60,
    )


def _classify_openai_error(error_msg: str) -> str:
    e = error_msg.lower()
    if "429" in error_msg:
        if "quota" in e or "billing" in e or "exceeded" in e or "insufficient" in e:
            return "API billing quota exhausted. Add credits to your OpenAI account."
        return "Rate limit hit. Please wait 30 seconds and try again."
    if "401" in error_msg or "incorrect api key" in e or "invalid api key" in e:
        return "API key is invalid or expired. Update OPENAI_API_KEY in your .env file."
    if "403" in error_msg:
        return "API access denied. Check your account permissions."
    if "timeout" in e or "timed out" in e:
        return "Request timed out. Please try again."
    if "503" in error_msg or "502" in error_msg or "unavailable" in e:
        return "AI service is temporarily down. Please try again in a moment."
    if "context_length" in e or "maximum context" in e:
        return "Query is too long. Please shorten your question."
    return f"AI service error: {error_msg[:120]}"


def invoke_with_fallback(llm, prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content:
            return content
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content:
            return content
        return "The AI could not generate a response. Please try rephrasing your question."
    except Exception as e:
        return _classify_openai_error(str(e))


async def astream_with_fallback(llm, prompt: str):
    try:
        full_content = ""
        async for chunk in llm.astream(prompt):
            if chunk.content:
                full_content += chunk.content
                yield chunk.content

        if not full_content.strip():
            response = llm.invoke(prompt)
            content = response.content.strip()
            if content:
                yield content
            else:
                yield "The AI could not generate a response. Please try rephrasing."

    except Exception as e:
        yield _classify_openai_error(str(e))