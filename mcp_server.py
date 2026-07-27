"""Personal MCP server: query Avinash's résumé and draft application material.

Not the public portfolio chatbot (see main.py, which serves site visitors via
Groq). This runs locally over stdio so subagents (or Avinash directly) can
call it as a tool: fetch the résumé, get writing guidance for a specific
task, then do the actual writing themselves — no separate LLM call here,
the calling agent already is one.
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

CONVERSATIONAL_GUIDE = """
Conversational tone, not a formal letter template:
- Skip "Dear Hiring Manager" / "To Whom It May Concern" / "I am writing to express my interest."
  Open like you'd open an email to someone you want to work with.
- Skip "In closing" / "Thank you for your consideration" sign-offs. End when you're done.
- It should read like Avinash explaining why he's a fit over coffee, not reciting a résumé.
"""


@mcp.tool()
def get_resume() -> str:
    """Avinash's full résumé/background — call this for any question about his experience, skills, or history that the other tools don't already cover."""
    return RESUME


@mcp.tool()
def cover_letter(company: str, role: str, job_description: str = "") -> str:
    """Get the résumé plus writing guidance for a conversational cover letter. Call this, then write the letter yourself following the returned instructions."""
    jd_block = f"\n\nJob description:\n{job_description}" if job_description else ""
    return f"""Using the résumé below, write a cover letter from Avinash for the \
{role} role at {company}.{jd_block}

{HUMAN_VOICE_GUIDE}
{CONVERSATIONAL_GUIDE}

Keep it under 350 words, 3-4 short paragraphs.

RÉSUMÉ:
{RESUME}"""


@mcp.tool()
def application_question(question: str, company: str = "", role: str = "") -> str:
    """Get the résumé plus writing guidance for a short application-form answer (e.g. "Why do you want to work at X?"). Call this, then write the answer yourself following the returned instructions."""
    where = f" for the {role} role at {company}" if company else ""
    return f"""Using the résumé below, answer this application question from \
Avinash's perspective{where}:

"{question}"

{HUMAN_VOICE_GUIDE}

Keep it to 2-4 sentences unless the question clearly needs more.

RÉSUMÉ:
{RESUME}"""


@mcp.tool()
def career_question(question: str, company: str = "", role: str = "", context: str = "") -> str:
    """Catch-all for career/job-search questions that aren't a cover letter or a single application-form answer: interview prep, salary talking points, recruiter/LinkedIn messages, follow-up emails, resume tailoring advice, etc. Call this, then respond yourself following the returned instructions."""
    where = f" for the {role} role at {company}" if company else ""
    ctx_block = f"\n\nAdditional context:\n{context}" if context else ""
    return f"""Using the résumé below, help Avinash with this{where}:

"{question}"{ctx_block}

{HUMAN_VOICE_GUIDE}

Match the response length and format to what's actually being asked for.

RÉSUMÉ:
{RESUME}"""


if __name__ == "__main__":
    mcp.run()
