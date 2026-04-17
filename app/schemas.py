from pydantic import BaseModel, Field, model_validator


class IpoInput(BaseModel):
    nii: float = Field(ge=0, description="NII subscription (x)")
    qib: float = Field(ge=0, description="QIB subscription (x)")
    retail: float = Field(ge=0, description="Retail subscription (x)")
    total: float = Field(gt=0, description="Total subscription (x). Must be > 0.")
    year: int = Field(ge=2000, le=2100, description="IPO year")
    issue_amount: float = Field(gt=0, description="Issue amount (Rs. crores)")
    price_band_high: float = Field(gt=0, description="Upper price band (Rs.)")
    price_band_low: float = Field(gt=0, description="Lower price band (Rs.)")
    gmp: float | None = Field(default=None, description="Grey market premium on close. Null if unavailable.")

    @model_validator(mode="after")
    def _validate_price_bands(self):
        if self.price_band_high < self.price_band_low:
            raise ValueError("price_band_high must be >= price_band_low")
        return self


class PredictRequest(BaseModel):
    ipos: list[IpoInput] = Field(min_length=1, description="IPOs listed on a given day")


class IpoAllocation(BaseModel):
    probability: float = Field(description="Predicted probability of listing gain exceeding threshold")
    allocation_weight: float = Field(description="Portfolio allocation weight")


class PredictResponse(BaseModel):
    allocations: list[IpoAllocation]
