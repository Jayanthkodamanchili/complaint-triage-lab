# NEW CODE START
"""The answer sheet every LLM must fill in (Pydantic schema)."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# The only answers allowed in each "box" of the answer sheet
Product = Literal[
    "credit_reporting", "debt_collection", "checking_savings", "credit_card", "mortgage"
]
Urgency = Literal["low", "medium", "high"]


class TriageResult(BaseModel):
    product: Product = Field(
        description="Which of the 5 products the complaint is about"
    )
    issue: str = Field(description="The main problem in 3 to 8 words")
    urgency: Urgency = Field(
        description=(
            "high = money lost or at risk right now, foreclosure, fraud or identity theft, "
            "or a legal threat; medium = an ongoing problem that needs fixing soon; "
            "low = a question, a small error, or a problem that is already over"
        )
    )
    summary: str = Field(description="One plain sentence describing the complaint")


if __name__ == "__main__":
    good = '{"product": "mortgage", "issue": "escrow payment raised", "urgency": "medium", "summary": "The customer says the bank raised the escrow payment without notice."}'
    bad = '{"product": "student_loan", "issue": "late fee", "urgency": "high", "summary": "The customer was charged a late fee."}'

    print("GOOD answer sheet:")
    print(TriageResult.model_validate_json(good))

    print("BAD answer sheet:")
    try:
        TriageResult.model_validate_json(bad)
    except ValidationError as error:
        print("REJECTED:", error.errors()[0]["msg"])
# NEW CODE END
