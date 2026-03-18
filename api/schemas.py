"""
Pydantic schemas for the FastAPI no-show prediction API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ServiceType(str, Enum):
    HAIRCUT = "Haircut"
    COLOR = "Color"
    KERATIN = "Keratin"
    FACIAL = "Facial"
    MANICURE = "Manicure"
    PEDICURE = "Pedicure"
    WAXING = "Waxing"
    BRIDAL = "Bridal"


class Branch(str, Enum):
    SCIENCE_CITY = "Science City"
    MEMNAGAR = "Memnagar"
    SINDHU_BHAVAN_ROAD = "Sindhu Bhavan Road"
    SABARMATI = "Sabarmati"
    CHANDKHEDA = "Chandkheda"


class PaymentMethod(str, Enum):
    ONLINE_PREPAID = "Online Prepaid"
    CARD_ON_ARRIVAL = "Card on Arrival"
    CASH = "Cash"
    UPI = "UPI"


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BookingRequest(BaseModel):
    """Single booking prediction request."""
    service_type: ServiceType = Field(..., description="Type of salon service")
    branch: Branch = Field(..., description="Salon branch location")
    booking_lead_time_hours: int = Field(..., ge=0, le=720, description="Hours between booking and appointment")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of appointment")
    payment_method: PaymentMethod = Field(..., description="Payment method")
    past_visit_count: int = Field(0, ge=0, le=100, description="Customer's past visit count")
    past_cancellation_count: int = Field(0, ge=0, le=50, description="Past cancellation count")
    past_noshow_count: int = Field(0, ge=0, le=20, description="Past no-show count")
    service_duration_mins: int = Field(60, ge=10, le=400, description="Service duration in minutes")
    staff_id: str = Field("S01", description="Staff member ID")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "service_type": "Haircut",
                    "branch": "Memnagar",
                    "booking_lead_time_hours": 24,
                    "day_of_week": 2,
                    "hour_of_day": 14,
                    "payment_method": "Online Prepaid",
                    "past_visit_count": 5,
                    "past_cancellation_count": 1,
                    "past_noshow_count": 0,
                    "service_duration_mins": 45,
                    "staff_id": "S05",
                }
            ]
        }
    }


class BatchBookingRequest(BaseModel):
    """Batch prediction request."""
    bookings: List[BookingRequest] = Field(..., description="List of bookings to predict")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    noshow_probability: float = Field(..., description="Predicted probability of no-show")
    risk_tier: RiskTier = Field(..., description="Risk classification tier")
    risk_factors: List[str] = Field(..., description="Top risk factors for this booking")
    recommended_action: str = Field(..., description="Recommended action to mitigate risk")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    n_features: int
    feature_names: List[str]
