"""Personal MCP server: query Avinash's résumé and draft application material.

Not the public portfolio chatbot (see main.py, which serves site visitors via
Groq). This runs locally over stdio so Avinash can ask Claude about his own
background and get cover letters / application-question answers written in
his voice.
"""
from mcp.server.fastmcp import FastMCP

from persona import RESUME

mcp = FastMCP("avinash-persona")

HUMAN_VOICE_GUIDE = """
Write like a person typed this in one sitting, not like an AI generated it:
- No "I am excited to leverage my skills" / "passionate about" / "proven track record" stock phrases.
- No em-dash-heavy rhythm, no "It's not just X, it's Y" constructions.
- Vary sentence length. Let a sentence be short.
- Use concrete details from the résumé (real companies, real numbers) instead of vague claims.
- First person, direct, a little informal is fine. No corporate buzzword salad.
- Say what's true and stop. Don't pad.
"""


@mcp.resource("resume://avinash")
def resume() -> str:
    """Avinash's full résumé/background — source of truth for all answers."""
    return RESUME


@mcp.prompt()
def cover_letter(company: str, role: str, job_description: str = "") -> str:
    """Draft a cover letter for a specific company and role."""
    jd_block = f"\n\nJob description:\n{job_description}" if job_description else ""
    return f"""Using the résumé below, write a cover letter from Avinash for the \
{role} role at {company}.{jd_block}

{HUMAN_VOICE_GUIDE}

Keep it under 350 words, 3-4 paragraphs, simple greeting and sign-off (no \
boilerplate letterhead).

RÉSUMÉ:
{RESUME}"""


@mcp.prompt()
def application_question(question: str, company: str = "", role: str = "") -> str:
    """Answer an application question, e.g. 'Why do you want to work at X?'"""
    where = f" for the {role} role at {company}" if company else ""
    return f"""Using the résumé below, answer this application question from \
Avinash's perspective{where}:

"{question}"

{HUMAN_VOICE_GUIDE}

Keep it to 2-4 sentences unless the question clearly needs more.

RÉSUMÉ:
{RESUME}"""


if __name__ == "__main__":
    mcp.run()
