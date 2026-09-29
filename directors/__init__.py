"""AI video director package.

Directors decide WHAT visual edit should happen and WHEN.
The renderer is responsible only for executing that plan.
"""

from .models import DirectorAction, DirectorPlan
from .router import route_director

__all__ = ["DirectorAction", "DirectorPlan", "route_director"]
