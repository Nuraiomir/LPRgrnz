from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="Local OCRM API",
    description="Local mock backend for LPR/OCRM integration",
    version="1.0.0",
)


# ============================================================
# TEST DATABASE
# ============================================================

VEHICLES = {
    "322BDM02": {
        "plate": "322BDM02",

        "client": {
            "name": "Иванов Иван Иванович",
            "iin": "000000000001",
            "phone": "+7 700 000 0001",
        },

        "loan": {
            "loan_id": "DBZ-000001",
            "status": "Просрочен",
            "overdue_days": 245,
        },

        "vehicle": {
            "brand": "Toyota",
            "model": "Camry",
            "year": 2020,
            "color": "Черный",
        },

        "pledge": {
            "status": "Залоговый",
            "collateral_id": "COL-000001",
        },

        "organization": {
            "branch": "Алматы",
            "manager": "Петров П.П.",
        },
    },

    "118BKY02": {
        "plate": "118BKY02",

        "client": {
            "name": "Петрова Анна Сергеевна",
            "iin": "000000000002",
            "phone": "+7 700 000 0002",
        },

        "loan": {
            "loan_id": "DBZ-000002",
            "status": "Просрочен",
            "overdue_days": 127,
        },

        "vehicle": {
            "brand": "Hyundai",
            "model": "Tucson",
            "year": 2021,
            "color": "Белый",
        },

        "pledge": {
            "status": "Залоговый",
            "collateral_id": "COL-000002",
        },

        "organization": {
            "branch": "Алматы",
            "manager": "Сидоров С.С.",
        },
    },

    "099JAX02": {
        "plate": "099JAX02",

        "client": {
            "name": "Садыков Нурлан Ерланович",
            "iin": "000000000003",
            "phone": "+7 700 000 0003",
        },

        "loan": {
            "loan_id": "DBZ-000003",
            "status": "Просрочен",
            "overdue_days": 89,
        },

        "vehicle": {
            "brand": "Kia",
            "model": "Sportage",
            "year": 2019,
            "color": "Серый",
        },

        "pledge": {
            "status": "Залоговый",
            "collateral_id": "COL-000003",
        },

        "organization": {
            "branch": "Алматы",
            "manager": "Ахметов А.А.",
        },
    },

    "375BWS02": {
        "plate": "375BWS02",

        "client": {
            "name": "Ким Алексей Викторович",
            "iin": "000000000004",
            "phone": "+7 700 000 0004",
        },

        "loan": {
            "loan_id": "DBZ-000004",
            "status": "Просрочен",
            "overdue_days": 310,
        },

        "vehicle": {
            "brand": "Toyota",
            "model": "RAV4",
            "year": 2022,
            "color": "Белый",
        },

        "pledge": {
            "status": "Залоговый",
            "collateral_id": "COL-000004",
        },

        "organization": {
            "branch": "Алматы",
            "manager": "Иванова И.И.",
        },
    },

    "184BST02": {
        "plate": "184BST02",

        "client": {
            "name": "Ахметов Данияр Маратович",
            "iin": "000000000005",
            "phone": "+7 700 000 0005",
        },

        "loan": {
            "loan_id": "DBZ-000005",
            "status": "Просрочен",
            "overdue_days": 61,
        },

        "vehicle": {
            "brand": "Lexus",
            "model": "RX 350",
            "year": 2020,
            "color": "Черный",
        },

        "pledge": {
            "status": "Залоговый",
            "collateral_id": "COL-000005",
        },

        "organization": {
            "branch": "Алматы",
            "manager": "Касымов К.К.",
        },
    },
}


# ============================================================
# VISIT STORAGE
# ============================================================

VISITS = []


# ============================================================
# REQUEST MODELS
# ============================================================

class VisitRequest(BaseModel):

    plate: str

    result: Optional[str] = None

    comment: Optional[str] = None

    latitude: Optional[float] = None

    longitude: Optional[float] = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Local OCRM API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "database": "local-test-data",
        "vehicles": len(VEHICLES),
        "visits": len(VISITS),
    }


# ============================================================
# SEARCH VEHICLE BY PLATE
# ============================================================

@app.get("/api/vehicle/{plate}")
def get_vehicle(plate: str):

    plate = (
        plate
        .upper()
        .strip()
        .replace(" ", "")
        .replace("-", "")
    )

    vehicle = VEHICLES.get(
        plate
    )

    if vehicle is None:

        raise HTTPException(
            status_code=404,
            detail={
                "message": "Vehicle not found",
                "plate": plate,
            },
        )

    return {
        "found": True,
        "data": vehicle,
    }


# ============================================================
# SEARCH ALL VEHICLES
# ============================================================

@app.get("/api/vehicles")
def get_all_vehicles():

    return {
        "count": len(VEHICLES),
        "vehicles": list(
            VEHICLES.values()
        ),
    }


# ============================================================
# CREATE VISIT
# ============================================================

@app.post("/api/visit")
def create_visit(
    request: VisitRequest
):

    plate = (
        request.plate
        .upper()
        .strip()
        .replace(" ", "")
        .replace("-", "")
    )


    vehicle_exists = (
        plate in VEHICLES
    )


    visit = {

        "id": len(VISITS) + 1,

        "created_at": (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        ),

        "plate": plate,

        "vehicle_found": vehicle_exists,

        "result": request.result,

        "comment": request.comment,

        "latitude": request.latitude,

        "longitude": request.longitude,
    }


    VISITS.append(
        visit
    )


    return {
        "success": True,
        "visit": visit,
    }


# ============================================================
# GET VISITS
# ============================================================

@app.get("/api/visits")
def get_visits():

    return {
        "count": len(VISITS),
        "visits": VISITS,
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "ocrm_backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )