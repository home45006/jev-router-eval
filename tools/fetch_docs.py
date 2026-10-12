"""在 GitHub Actions 上抓取 DeepSeek 官方 API 文档的相关页面，转成纯文本打印到日志（本地开发环境访问不了文档站）。

    python tools/fetch_docs.py
"""
import re
import sys
import urllib.request
from html.parser import HTMLParser

BASE = "https://api-docs.deepseek.com"
KEYWORDS = ("anthropic", "search", "tool", "thinking", "pricing", "claude", "function", "reasoning", "models", "news")
ALWAYS = ["/zh-cn/", "/zh-cn/guides/anthropic_api", "/zh-cn/quick_start/agent_integrations/claude_code",
          "/zh-cn/quick_start/pricing", "/zh-cn/guides/thinking_mode", "/zh-cn/guides/function_calling"]


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 docs-fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


class Text(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "svg"}

    def __init__(self):
        super().__init__()
        self.out, self.skip, self.in_main = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "tr", "pre", "br", "div"):
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("#" * int(tag[1]) + " ")
        if tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def text_of(html: str) -> str:
    m = re.search(r"<article.*?</article>", html, re.S) or re.search(r"<main.*?</main>", html, re.S)
    p = Text()
    p.feed(m.group(0) if m else html)
    t = "".join(p.out)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def main():
    urls = []
    try:
        sm = get(BASE + "/sitemap.xml")
        all_urls = re.findall(r"<loc>(.*?)</loc>", sm)
        print("## sitemap 里的中文页面\n" + "\n".join(u for u in all_urls if "/zh-cn" in u) + "\n")
        urls = [u for u in all_urls if "/zh-cn" in u and any(k in u.lower() for k in KEYWORDS)]
    except Exception as e:
        print(f"sitemap 失败：{e}")
    for p in ALWAYS:
        if BASE + p not in urls:
            urls.append(BASE + p)
    for u in urls:
        try:
            t = text_of(get(u))
            print(f"\n\n==================== {u} ====================\n{t[:12000]}")
        except Exception as e:
            print(f"\n\n==================== {u} ====================\n失败：{e}")


if __name__ == "__main__":
    sys.exit(main())
