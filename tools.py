# version 1 imports
import re
import urllib.request as ur

import types
import textwrap

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# version 2 imports
import os
from pathlib import Path
import shutil

tools_patterns = {
    # version 1 patterns
    r"<calculate(?:\s+(.*?))?>": "calculator",

    r"<date(?:\s+(.*?))?>": "get_date",
    r"<time(?:\s+(.*?))?>": "get_time",

    r"<search(?:\s+(.*?))?>": "search_web",
    r"<fix_tags>": "fix_html_tags",
    r"<layout(?:\s+(.*?))?>": "change_layout",
    # version 2 patterns

    r"<read(?:\s+(.*?))?>": "read_file",
    r"<mkdir(?:\s+(.*?))?>": "make_directory",
    r"<dir(?:\s+(.*?))?>": "get_available_files",
    r"<mkfile(?:\s+(.*?))?>": "create_file",
    r"<del(?:\s+(.*?))?>": "delete_path",
    }

# version 1 code part
def check_for_toolcall(answer: str) -> bool:
    for pattern in tools_patterns.keys():
        match = re.search(pattern, answer)
        if match:
            return True
    return False

def _default_get_role(msg):
    return msg["role"]

def _default_get_content(msg):
    return msg["content"]

def _default_set_content(msg, content):
    msg["content"] = content
    return msg

def _default_make_message(role, content):
    return {"role": role, "content": content}

def connect(
    input_data,
    generate_function,
    *,
    max_iterations: int = 5,
    system_role: str = "system",
    assistant_role: str = "assistant",
    get_role=_default_get_role,
    get_content=_default_get_content,
    set_content=_default_set_content,
    make_message=_default_make_message,
    call_generate=None,
    is_async: bool = False,
):
    """
    Universal input for any AI backend
    input_data:
    - str -> one-time request, returns the final response string
    - list -> message history, returns an updated history list
    - generator/func -> stream generation, returns a chunk generator

    generate_function: what actually triggers your model. The signature depends
    on call_generate (see below)
    By default, it expects generate_function(history) -> str.

    If your message/generator call format is non-standard, pass:
    get_role(msg) -> str - get the role from the message
    get_content(msg) -> str - get the text from the message
    set_content(msg, content) - write text to the message (for the stream system)
    make_message(role, content) - create a new message in your format
    call_generate(generate_function, history) -> str - how exactly to call your model
    (for example, a lambda that unpacks history into a different format, adds
    await, additional parameters, etc.)
    max_iterations - how many times in a row the tools can be called
    system_role / assistant_role - names of roles in your history

    is_async=True - if generate_function / call_generate return awaitable
    (then use connect in conjunction with await, or pass your own
    call_generate, which wraps the asynchronous call synchronously, for example via
    asyncio.run).
    """
    if call_generate is None:
        call_generate = lambda gen_func, history: gen_func(history)

    common_kwargs = dict(
        max_iterations=max_iterations,
        system_role=system_role,
        assistant_role=assistant_role,
        get_role=get_role,
        get_content=get_content,
        set_content=set_content,
        make_message=make_message,
        call_generate=call_generate,
        is_async=is_async,
    )

    if isinstance(input_data, (types.GeneratorType, types.FunctionType)) or callable(input_data):
        return system_stream(input_data)
    elif isinstance(input_data, list):
        return system(input_data, generate_function, **common_kwargs)
    elif isinstance(input_data, str):
        mock_history = [make_message(assistant_role, input_data)]
        res_history = system(mock_history, generate_function, **common_kwargs)
        return get_content(res_history[-1])
    return input_data

def system(
    history: list,
    generate_function,
    *,
    max_iterations: int = 5,
    system_role: str = "system",
    assistant_role: str = "assistant",
    get_role=_default_get_role,
    get_content=_default_get_content,
    set_content=_default_set_content,
    make_message=_default_make_message,
    call_generate=None,
    is_async: bool = False,
) -> list:
    if call_generate is None:
        call_generate = lambda gen_func, hist: gen_func(hist)

    if not history:
        return history

    iteration = 0

    while (
        iteration < max_iterations
        and any(re.search(pat, get_content(history[-1])) for pat in tools_patterns.keys())
    ):
        raw_ai_output = get_content(history[-1])

        found_tag = None
        for pat in tools_patterns.keys():
            match = re.search(pat, raw_ai_output)
            if match:
                found_tag = match.group(0)
                break

        if not found_tag:
            break

        tool_result = use_all(found_tag)
        result_block = f"{found_tag} -> {tool_result}"

        history.append(make_message(system_role, result_block))

        next_ai_output = call_generate(generate_function, history)
        if is_async and hasattr(next_ai_output, "__await__"):
            raise TypeError(
                "generate_function returned awaitable, but is_async wasn't handled."
                "Pass call_generate, which unwraps the coroutine itself"
                "(for example, via asyncio.run(...) or your own event loop)."
            )
        history.append(make_message(assistant_role, next_ai_output))
        iteration += 1

    return history

def system_stream(
    generator,
    *,
    call_generator=None,
    tag_pattern=r"(</?([a-zA-Z0-9_]+)(?:\s+[^>]*)?>)",
):
    """
    A general-purpose stream handler.
    generator: a text chunk generator, OR a function without required
    arguments that returns such a generator (e.g.
    `lambda: client.stream(prompt)`).

    If your generation function requires arguments (prompt, history,
    additional parameters), pass them through call_generator, for example:
    system_stream(lambda: my_stream_call(prompt, history))
    or directly pass a ready-made chunk generator/iterator to generator.

    tag_pattern - if your tags differ from `<name attrs>` /
    `</name>`, you can pass your own regex with the same group structure.
    """
    if callable(generator) and not isinstance(generator, types.GeneratorType):
        generator = call_generator(generator) if call_generator else generator()

    buffer = ""
    inside_tag = False

    for chunk in generator:
        buffer += chunk

        if "<" in buffer and not inside_tag:
            clean_part, tag_start = buffer.split("<", 1)
            if clean_part:
                yield clean_part
            buffer = "<" + tag_start
            inside_tag = True

        if inside_tag and ">" in buffer:
            tag_match = re.search(tag_pattern, buffer)

            if tag_match:
                full_tag = tag_match.group(1)
                is_tool = any(re.search(pat, full_tag) for pat in tools_patterns.keys())

                if is_tool:
                    tool_result = use_all(full_tag)
                    result_block = f"{full_tag} -> {tool_result}"
                    yield result_block

                    buffer = buffer.replace(full_tag, "", 1)
                    inside_tag = False
                else:
                    yield full_tag
                    buffer = buffer.replace(full_tag, "", 1)
                    if "<" not in buffer:
                        inside_tag = False

        if not inside_tag and buffer:
            yield buffer
            buffer = ""

    if buffer:
        yield fix_html_tags(buffer)

def use_all(answer: str) -> str:
    '''
    Takes a text(str) and returns the text with 
    all detected tool calls applied.
    '''
    try:
        for pattern in tools_patterns.keys():
            match = re.search(pattern, answer)
            if match:
                func_name = tools_patterns.get(pattern)
                func = globals()[func_name]
                answer = func(answer)
        return answer
    except Exception:
        pass

def calculator(answer: str) -> str:
    pattern = list(tools_patterns.keys())[0]
    matches = re.findall(pattern, answer)

    for match in matches:
        result = str(eval(match))
        answer = answer.replace(f"<calculate {match}>", result)
    return answer

def get_date(answer: str) -> str:

    def match_handler(match):
            params = match.group(1) if match.group(1) else ""
            params = params.strip()
            
            now = datetime.today()
    
            if not params:
                return now.strftime("%d.%m.%Y")
            if params in ['y', 'M', 'd']:
                if params == 'y':
                    return str(now.year)
                elif params == 'M':
                    return str(now.month)
                elif params == 'd':
                    return str(now.day)
                
            shifts = re.findall(r'([+-]?\d+)([dMy])', params)
    
            if not shifts:
                return now.strftime("%d.%m.%Y")
                
            total_days = 0
            total_months = 0
            total_years = 0
            
            for value, unit in shifts:
                value = int(value)
                if unit == 'y':
                    total_years += value
                elif unit == 'M':
                    total_months += value
                elif unit == 'd':
                    total_days += value
                    
            future_date = now + relativedelta(
                years=total_years, months=total_months,
                days=total_days
            )
            return future_date.strftime("%d.%m.%Y")
    
    pattern = list(tools_patterns.keys())[1]
    return re.sub(pattern, match_handler, answer)
    
def get_time(text: str) -> str:
    
    def match_handler(match):
        params = match.group(1) if match.group(1) else ""
        params = params.strip()
        
        now = datetime.now()

        if not params:
            return now.strftime("%H:%M:%S")
        if params in ['h', 'm', 's']:
            if params == 'h':
                return now.strftime("%H")
            elif params == 'm':
                return now.strftime("%M")
            elif params == 's':
                return now.strftime("%S")
            
        shifts = re.findall(r'([+-]?\d+)([hm])', params)

        if not shifts:
            return now.strftime("%H:%M:%S")
            
        total_hours = 0
        total_minutes = 0
        
        for value, unit in shifts:
            value = int(value)
            if unit == 'h':
                total_hours += value
            elif unit == 'm':
                total_minutes += value
                
        future_time = now + timedelta(hours=total_hours, minutes=total_minutes)
        return future_time.strftime("%H:%M:%S")

    pattern = list(tools_patterns.keys())[2]
    return re.sub(pattern, match_handler, text)

def change_layout(answer: str) -> str:

    en_chars = "qwertyuiop[]asdfghjkl;'zxcvbnm,.QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>"
    ru_chars = "йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ"
    en_to_ru = str.maketrans(en_chars, ru_chars)
    ru_to_en = str.maketrans(ru_chars, en_chars)

    def match_handler(match):
        params = match.group(1) if match.group(1) else ""
        params = params.strip()
        if not params:
            return params

        fixed = params.translate(en_to_ru) if (params[0] in en_chars) else params.translate(ru_to_en)
        return fixed

    pattern = list(tools_patterns.keys())[5]
    return re.sub(pattern, match_handler, answer)


def search_web(answer: str) -> str:
    def match_handler(match):
        query = match.group(1) if match.group(1) else ""
        query = query.strip()
        query = use_all(query)

        if not query:
            return "[search: empty request]"

        try:
            try:
                from ddgs import DDGS #type:ignore
            except ImportError:
                from duckduckgo_search import DDGS #type:ignore

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
        except Exception as e:
            return f"search error: {e}"

        if not results:
            return f"Nothing found for the search «{query}»"

        links = []
        lines = []
        for r in results:
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            links.append(href)
            lines.append(f"Information:\n{body[30:80] + "..." if len(body) >= 80 else body[:50] + "..."}")
        texts = _deep_search(links)
        result = textwrap.shorten(f'{" ".join(lines)} Details: {texts}', width=800, placeholder="...")
        return result

    pattern = list(tools_patterns.keys())[3]
    return re.sub(pattern, match_handler, answer)

def _deep_search(urls: list) -> str:
    import urllib.parse as up
    from bs4 import BeautifulSoup as bs
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    texts = []
    need_one = int(300 / len(urls))
    start = 100
    end = start + need_one

    for url in urls:
        safe_url = up.quote(url, safe=':/?&=')

        req = ur.Request(
            safe_url,
            headers=headers
        )
        try:
            with ur.urlopen(req, timeout=3) as response:
                raw = response.read().decode("utf-8")
            soup = bs(raw, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            texts.append(text[start:end] + "...")
            start += int(len(text) / need_one)
            end = start + end
        except Exception as e:
            texts.append("The page is unavalaible")
    return " ".join(texts)

def fix_html_tags(answer: str) -> str:
    answer = answer.replace("<fix_tags>", "")
    single_tags = {"img", "br", "input", "hr", "meta", "link", "area", "base", "col", "embed"}

    stack = []
    fixed_chunks = []
    last_idx = 0
    
    for match in re.finditer(r"<[^>]+>", answer):
        start, end = match.span()
        tag_text = match.group(0)
        
        fixed_chunks.append(answer[last_idx:start])
        last_idx = end
        
        is_closing = tag_text.startswith("</")
        is_self_closing = tag_text.endswith("/>")

        name_match = re.search(r"<\/*([a-zA-Z1-6]+)", tag_text)
        if not name_match:
            fixed_chunks.append(tag_text)
            continue
            
        tag_name = name_match.group(1).lower()
        
        if tag_name in single_tags or is_self_closing:
            fixed_chunks.append(tag_text)
            continue
            
        if not is_closing:
            stack.append(tag_name)
            fixed_chunks.append(tag_text)
        else:
            if stack:
                if stack[-1] == tag_name:
                    stack.pop()
                    fixed_chunks.append(tag_text)
                else:
                    correction = ""
                    while stack and stack[-1] != tag_name:
                        missed_tag = stack.pop()
                        correction += f"</{missed_tag}>"
                    
                    if stack and stack[-1] == tag_name:
                        stack.pop()
                        
                    fixed_chunks.append(correction + tag_text)
            else:
                pass

    fixed_chunks.append(answer[last_idx:])
    end_correction = ""
    while len(stack) != 0:
        missed_tag = stack.pop()
        end_correction += f"</{missed_tag}>"
        
    return "".join(fixed_chunks) + end_correction

# version 2 code part
WORKSPACE_DIR = Path("Workspace").resolve()
WORKSPACE_DIR.mkdir(exist_ok=True)

# get the full path to file
def get_safe_path(path: str) -> Path:
    target = (WORKSPACE_DIR / path.strip()).resolve()
    if not target.is_relative_to(WORKSPACE_DIR):
        raise(PermissionError("Acecess Denied: AI model tried to go beyond 'Workspace'!"))
    return target

def read_file(answer: str) -> str:

    def match_handler(match):
        try:
            path = match.group(1) if match.group(1) else ""
            safe_path = get_safe_path(path)
            if not safe_path.is_file():
                return "this file does not exist"
            with open(safe_path, "r", encoding="utf-8") as f:
                return f"contents of file {path}: {f.read()}"
        except PermissionError:
            return f"access to {path} is denied"
        
        except Exception:
            return f"error reading file {path}"

    pattern = list(tools_patterns.keys())[6]
    return re.sub(pattern, match_handler, answer)

def make_directory(answer: str) -> str:

    def match_handler(match):
        try:
            path = match.group(1)
            safe_path = get_safe_path(path)
            safe_path.mkdir(parents=True, exist_ok=True)
            return f"folder {path} successfully created"

        except PermissionError:
            return f"access to {path} is denied"
        except Exception:
            return f"error creating folder {path}" 

    pattern = list(tools_patterns.keys())[7]
    return re.sub(pattern, match_handler, answer)

def get_available_files(answer: str) -> str:

    def match_handler(match):
        try:
            path = match.group(1) if match.group(1) else ""
            if not path:
                items = os.listdir(WORKSPACE_DIR)
                path = WORKSPACE_DIR
                simple = path.name
            else:
                items = os.listdir(WORKSPACE_DIR / path)
                path = WORKSPACE_DIR / path
                simple = Path(path).relative_to(WORKSPACE_DIR)
            if not items:
                return f"the {simple} directory is empty"
            result = f"available objects in {simple}:"
            for item in items:
                item_path = Path(path / item)
                if item_path.is_dir():
                    result += f"folder {item}, "
                else:
                    result += f"file {item}, "
            return result
        except PermissionError:
            return "access denied"
        except Exception:
            return f"error getting available files in {path}"
    pattern = list(tools_patterns.keys())[8]
    return re.sub(pattern, match_handler, answer)

def create_file(answer: str) -> str:

    def match_handler(match):
        try:
            params = match.group(1)
            path, body = params.strip().split(" ", 1)
            safe_path = get_safe_path(path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(body, encoding="utf-8")
            return f"file {path} successfully created"

        except PermissionError:
            return "access denied"
        except Exception:
            return f"error creating file {path}"

    pattern = list(tools_patterns)[9]
    return re.sub(pattern, match_handler, answer)

def delete_path(answer: str) -> str:

    def match_handler(match):
        try:
            path = match.group(1)
            safe_path = get_safe_path(path)
            if safe_path == WORKSPACE_DIR:
                return "access denied"
            if not safe_path.exists():
                return "this path does not exist"
            if safe_path.is_dir():
                shutil.rmtree(safe_path)
                return f"folder {path} has been completely deleted"
            else:
                os.remove(safe_path)
                return f"file {path} was deleted"

        except PermissionError:
            return "access denied"
        except Exception:
            return "error deleting object"

    pattern = list(tools_patterns.keys())[10]
    return re.sub(pattern, match_handler, answer)