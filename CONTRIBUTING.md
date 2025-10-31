# 🤝 Contributing Guide

Thank you for your interest in contributing to the Community Flood and Heat Risk Analyzer! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive Behavior**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable Behavior**:
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/community-flood-heat-analyzer.git
cd community-flood-heat-analyzer
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install development tools
# pip install black pylint mypy pytest
# (Note: Test suite is in development - manual testing currently used)
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

## Development Process

### 1. Make Your Changes

- Write clear, concise code
- Follow the code standards (see below)
- Add comments for complex logic
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run the application locally
streamlit run app.py

# Manual testing: Test all affected features in the UI

# Optional code formatting (if tools installed)
# black *.py
# pylint *.py
```

### 3. Commit Your Changes

```bash
git add .
git commit -m "type: brief description

Detailed explanation of what changed and why.

Fixes #issue_number"
```

**Commit Message Format**:
```
type: subject line (max 50 chars)

Detailed description (wrap at 72 chars)
- What changed
- Why it changed
- Any breaking changes

References: #123, #456
```

**Commit Types**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

**Formatting**:
- Line length: 100 characters (not 79)
- Indentation: 4 spaces (no tabs)
- Use Black formatter for automatic formatting

**Naming Conventions**:
```python
# Classes: PascalCase
class EnhancedRiskModel:
    pass

# Functions/methods: snake_case
def calculate_risk_score():
    pass

# Constants: UPPER_CASE
MAX_RETRIES = 3

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

**Imports**:
```python
# Standard library
import os
from datetime import datetime

# Third-party
import pandas as pd
import numpy as np

# Local
from database import User
from nlp_analyzer import NewsTextAnalyzer
```

**Docstrings**:
```python
def analyze_risk(features: list, nlp_score: float) -> dict:
    """
    Analyze flood and heat risk based on features and NLP score.
    
    Args:
        features: List of 19 environmental features
        nlp_score: NLP-derived risk score (0.0-1.0)
    
    Returns:
        dict: Risk prediction with keys:
            - risk_level: 'Low', 'Medium', or 'High'
            - risk_score: Float between 0.0 and 1.0
            - probabilities: Dict of class probabilities
    
    Raises:
        ValueError: If features list has incorrect length
    """
    pass
```

**Type Hints**:
```python
from typing import List, Dict, Optional, Tuple

def process_articles(
    articles: List[Dict[str, str]], 
    risk_type: str = 'flood'
) -> Optional[pd.DataFrame]:
    """Process news articles for risk analysis."""
    pass
```

### Code Organization

**File Structure**:
- One class per file (preferred)
- Related functions can be grouped
- Maximum file length: 500 lines

**Function Length**:
- Maximum 50 lines per function
- If longer, refactor into smaller functions

**Complexity**:
- Maximum cyclomatic complexity: 10
- Use early returns to reduce nesting

### Error Handling

```python
# Good: Specific exceptions with helpful messages
try:
    data = fetch_api_data(url)
except requests.RequestException as e:
    logger.error(f"Failed to fetch data from {url}: {e}")
    return fallback_data()

# Bad: Bare except
try:
    data = fetch_api_data(url)
except:
    pass
```

### Performance Guidelines

1. **Database Queries**:
   - Use indexes on frequently queried columns
   - Batch operations when possible
   - Close sessions properly

2. **Caching**:
   - Use `@st.cache_data` for data fetching
   - Use `@st.cache_resource` for models
   - Set appropriate TTL values

3. **Memory**:
   - Clean up large objects when done
   - Avoid loading entire datasets if not needed
   - Use generators for large iterations

## Testing Guidelines

### Unit Tests

```python
import unittest
from enhanced_nlp_analyzer import EnhancedNewsTextAnalyzer

class TestNLPAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EnhancedNewsTextAnalyzer()
        self.sample_text = "Severe flooding reported in Manila..."
    
    def test_risk_score_range(self):
        """Test that risk scores are in valid range."""
        result = self.analyzer.calculate_enhanced_risk_score(
            self.sample_text, 'flood'
        )
        self.assertGreaterEqual(result['total_risk_score'], 0.0)
        self.assertLessEqual(result['total_risk_score'], 1.0)
    
    def test_entity_extraction(self):
        """Test named entity extraction."""
        entities = self.analyzer.extract_named_entities(self.sample_text)
        self.assertIn('locations', entities)
        self.assertIn('Manila', entities['locations'])
```

### Integration Tests

```python
def test_full_prediction_workflow():
    """Test complete prediction pipeline."""
    # Setup
    extractor = GeospatialFeatureExtractor()
    model = EnhancedRiskPredictionModel(model_type='ensemble')
    model.train()
    
    # Execute
    features = extractor.extract_all_features(14.5995, 120.9842)
    flood_factors = extractor.calculate_flood_risk_factors(features)
    heat_factors = extractor.calculate_heat_risk_factors(features)
    ml_features = extractor.prepare_ml_features(features, flood_factors, heat_factors)
    
    prediction = model.predict_flood_risk(ml_features, 0.5)
    
    # Verify
    assert prediction['risk_level'] in ['Low', 'Medium', 'High']
    assert 0 <= prediction['risk_score'] <= 1
    assert sum(prediction['probabilities'].values()) - 1.0 < 0.01
```

### Test Coverage

- Aim for >80% code coverage
- Test happy paths and edge cases
- Test error handling
- Mock external API calls

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/

# Run specific test file
python -m pytest tests/test_nlp.py

# Run specific test
python -m pytest tests/test_nlp.py::TestNLPAnalyzer::test_risk_score_range
```

## Pull Request Process

### 1. Before Submitting

**Checklist**:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear
- [ ] Changes are focused (one feature/fix per PR)

### 2. PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issues
Fixes #123
Related to #456

## Changes Made
- Added XYZ feature
- Fixed ABC bug
- Updated documentation

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed my own code
- [ ] Commented code in hard-to-understand areas
- [ ] Updated documentation
- [ ] Changes generate no new warnings
- [ ] Added tests that prove fix/feature works
- [ ] New and existing tests pass locally
```

### 3. Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Maintainer reviews code
3. **Requested Changes**: Address feedback
4. **Approval**: PR is approved
5. **Merge**: PR is merged to main branch

### 4. After Merge

- Pull latest changes: `git pull origin main`
- Delete your branch: `git branch -d feature/your-feature`
- Celebrate! 🎉

## Issue Reporting

### Bug Reports

Use this template:

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.11]
- Browser: [e.g., Chrome 120]

**Additional Context**
Any other context about the problem.
```

### Feature Requests

Use this template:

```markdown
**Feature Description**
A clear description of the feature.

**Problem It Solves**
What problem does this feature address?

**Proposed Solution**
How you think it should work.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

## Development Areas

### High Priority
- User authentication UI
- Alert monitoring system
- Email/SMS notifications
- Sentinel-1 SAR data integration

### Medium Priority
- Model retraining pipeline
- Performance optimization
- Enhanced testing suite
- API documentation

### Low Priority
- UI/UX improvements
- Additional visualization options
- More data sources
- Mobile responsiveness

## Questions?

- Check [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for technical details
- Review existing issues and PRs
- Ask in discussions (if available)

Thank you for contributing! 🙏
