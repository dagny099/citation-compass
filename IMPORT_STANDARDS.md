# Import Standards for Academic Citation Platform

## Standard Import Organization

All Python modules should follow this import order and style:

### 1. Standard Library Imports
```python
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
```

### 2. Third-Party Imports
```python
import pandas as pd
import numpy as np
import neo4j
from pydantic import BaseModel
```

### 3. Local Application Imports (Absolute)
```python
from src.models.paper import Paper
from src.services.ml_service import get_ml_service
from src.config.settings import settings
```

## Import Style Rules

### ✅ DO
- Use absolute imports starting with `src.`
- Group imports by type with blank lines between groups
- Sort imports alphabetically within each group
- Use explicit imports (`from module import SpecificClass`)

### ❌ DON'T
- Use relative imports (`from ..module import something`)
- Mix import styles within a file
- Import entire modules unless necessary (`import src.models`)
- Use `import *`

## Examples

### Good
```python
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from neo4j import GraphDatabase

from src.models.paper import Paper
from src.services.analytics_service import AnalyticsService
from src.config.settings import settings
```

### Bad
```python
from ..models.paper import Paper  # relative import
import src.models  # too broad
from src.services import *  # wildcard import
from neo4j import GraphDatabase
import pandas as pd  # mixing third-party with local
```

## Migration Strategy

Files should be updated to follow this standard during:
1. Bug fixes or feature additions
2. Dedicated refactoring sessions
3. New file creation

This ensures consistent, maintainable import patterns across the codebase.