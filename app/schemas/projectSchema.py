from typing import Optional
from pydantic import BaseModel


class ProjectBase(BaseModel):
    title : str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass 

