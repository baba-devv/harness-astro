from fastapi import APIRouter, Depends, HTTPException, status

from app.db.neo4j import get_driver
from app.models.user import ProfileUpdate, User, UserCreate
from app.repositories.user_repo import UserExistsError, UserRepository

router = APIRouter(prefix="/users", tags=["users"])


def get_repo() -> UserRepository:
    return UserRepository(get_driver())


@router.post("", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, repo: UserRepository = Depends(get_repo)):
    try:
        return await repo.create(data)
    except UserExistsError:
        raise HTTPException(status_code=409, detail="User already exists")


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str, repo: UserRepository = Depends(get_repo)):
    user = await repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=User)
async def update_user(user_id: str, data: ProfileUpdate, repo: UserRepository = Depends(get_repo)):
    user = await repo.update_profile(user_id, data)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
