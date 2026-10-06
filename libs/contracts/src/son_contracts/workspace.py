"""Contracts for the guided workspace, independent of business templates."""

from pydantic import BaseModel, Field, model_validator


class SetSceneRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    scene_code: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_scene(self) -> "SetSceneRequest":
        if self.name is not None:
            self.name = self.name.strip()
            if not self.name:
                raise ValueError("场景名称不能为空")
        if not self.name and not self.scene_code:
            raise ValueError("请填写场景名称")
        return self


class SceneSnapshot(BaseModel):
    code: str
    name: str
    description: str
