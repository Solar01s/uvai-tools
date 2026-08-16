import re
import urllib.request as ur

import types
import textwrap

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

tools_patterns = {
    r"<calculate(?:\s+(.*?))?>": "calculator",

    r"<date(?:\s+(.*?))?>": "get_date",
    r"<time(?:\s+(.*?))?>": "get_time",

    r"<search(?:\s+(.*?))?>": "search_web",
    r"<fix_tags>": "fix_html_tags",
    r"<layout(?:\s+(.*?))?>": "change_layout",
    }


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
    user_role: str = "user",
    assistant_role: str = "assistant",
    get_role=_default_get_role,
    get_content=_default_get_content,
    set_content=_default_set_content,
    make_message=_default_make_message,
    call_generate=None,
    is_async: bool = False,
):
    """
    Универсальный вход для любого ИИ-бэкенда.

    input_data:
        - str            -> одноразовый запрос, вернёт итоговую строку ответа
        - list            -> история сообщений, вернёт обновлённый список истории
        - generator/func  -> потоковая генерация, вернёт генератор чанков

    generate_function: то, что реально дёргает вашу модель. Сигнатура зависит
    от call_generate (см. ниже). По умолчанию ожидается generate_function(history) -> str.

    Если формат сообщений/вызова генератора у вас нестандартный - передайте:
        get_role(msg) -> str            - достать роль из сообщения
        get_content(msg) -> str         - достать текст из сообщения
        set_content(msg, content)       - записать текст в сообщение (для системы стрима)
        make_message(role, content)     - создать новое сообщение в вашем формате
        call_generate(generate_function, history) -> str
                                         - как именно вызывать вашу модель
                                           (например, лямбда, распаковывающая
                                           историю в другой формат, добавляющая
                                           await, доп. параметры и т.д.)
        max_iterations                  - сколько раз подряд можно вызывать тулы
        user_role / assistant_role      - имена ролей в вашей истории

    is_async=True - если generate_function / call_generate возвращают awaitable
    (тогда используйте connect в связке с await, либо передайте свой
    call_generate, оборачивающий асинхронный вызов синхронно, например через
    asyncio.run).
    """
    if call_generate is None:
        call_generate = lambda gen_func, history: gen_func(history)

    common_kwargs = dict(
        max_iterations=max_iterations,
        user_role=user_role,
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
    user_role: str = "user",
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

        history.append(make_message(user_role, result_block))

        next_ai_output = call_generate(generate_function, history)
        if is_async and hasattr(next_ai_output, "__await__"):
            raise TypeError(
                "generate_function вернул awaitable, но is_async обработан не был. "
                "Передайте call_generate, который сам разворачивает корутину "
                "(например через asyncio.run(...) или свой event loop)."
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
    Универсальный потоковый обработчик.

    generator: генератор чанков текста, ИЛИ функция без обязательных
    аргументов, которая возвращает такой генератор (например
    `lambda: client.stream(prompt)`).

    Если ваша функция генерации требует аргументы (промпт, историю,
    доп. параметры) - передайте их через call_generator, например:
        system_stream(lambda: my_stream_call(prompt, history))
    или сразу передайте готовый генератор/итератор чанков в generator.

    tag_pattern - если ваши теги отличаются от `<name attrs>` /
    `</name>`, можно передать свой регекс с той же структурой групп.
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
    try:
        for pattern in tools_patterns.keys():
            match = re.search(pattern, answer)
            if match:
                func_name = tools_patterns.get(pattern)
                func = globals()[func_name]
                answer = func(answer)
        return answer
    except Exception as e:
        raise(e)
        return answer

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
                    return now.year
                elif params == 'M':
                    return now.month
                elif params == 'd':
                    return now.day
                
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
    
    pattern = pattern = list(tools_patterns.keys())[1]
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
            return "[search: пустой запрос]"

        try:
            try:
                from ddgs import DDGS #type:ignore
            except ImportError:
                from duckduckgo_search import DDGS #type:ignore

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
        except Exception as e:
            return f"[search error: {e}]"

        if not results:
            return f"[search: по запросу «{query}» ничего не найдено]"

        links = []
        lines = []
        for r in results:
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            links.append(href)
            lines.append(f"Information:\n{body[30:80] + "..." if len(body) >= 80 else body[:50] + "..."}")
        texts = _deep_search(links)
        result = textwrap.shorten(f'{"\n".join(lines)}\nDetails:\n{texts}\n', width=800, placeholder="...")
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
            texts.append("Страница недоступна")
    return "\n".join(texts)

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
