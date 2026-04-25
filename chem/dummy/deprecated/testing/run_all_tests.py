"""
chem/dummy/testing/run_all_tests.py

Master test runner for all dummy module tests.
Run from project root: python -m chem.dummy.testing.run_all_tests
"""

import sys

from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    print("="*70)
    print(" DUMMY MODULE TEST SUITE")
    print("="*70)
    
    all_passed = True
    results = {}
    
    # Test DummyManager
    print("\n" + "~"*70)
    print(" Testing: dummy_manager.py")
    print("~"*70)
    try:
        from chem.dummy.testing.test_dummy_manager import run_all_tests as test_manager
        results["dummy_manager"] = test_manager()
        if not results["dummy_manager"]:
            all_passed = False
    except Exception as e:
        print(f"✗ Failed to run dummy_manager tests: {e}")
        import traceback
        traceback.print_exc()
        results["dummy_manager"] = False
        all_passed = False
    
    # Test dummy_utils
    print("\n" + "~"*70)
    print(" Testing: dummy_utils.py")
    print("~"*70)
    try:
        from chem.dummy.testing.test_dummy_utils import run_all_tests as test_utils
        results["dummy_utils"] = test_utils()
        if not results["dummy_utils"]:
            all_passed = False
    except Exception as e:
        print(f"✗ Failed to run dummy_utils tests: {e}")
        import traceback
        traceback.print_exc()
        results["dummy_utils"] = False
        all_passed = False
    
    # Test dummy_cleaning
    print("\n" + "~"*70)
    print(" Testing: dummy_cleaning.py")
    print("~"*70)
    try:
        from chem.dummy.testing.test_dummy_cleaning import run_all_tests as test_cleaning
        results["dummy_cleaning"] = test_cleaning()
        if not results["dummy_cleaning"]:
            all_passed = False
    except Exception as e:
        print(f"✗ Failed to run dummy_cleaning tests: {e}")
        import traceback
        traceback.print_exc()
        results["dummy_cleaning"] = False
        all_passed = False
    
    # Final summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    
    for module, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {module}: {status}")
    
    print("="*70)
    if all_passed:
        print(" ALL TESTS PASSED")
    else:
        print(" SOME TESTS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)