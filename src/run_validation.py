"""
Simplified Turtle Mapping Validation Script
Run this script to validate lingomap_rules.ttl
"""
# python run_validation.py
# python validate_mapping.py lingomap_rules.ttl

from validate_mapping import MappingValidator

def main():
    """Main function"""
    mapping_file = "lingomap_rules.ttl"
    
    print("🔍 LingoMap Turtle Mapping Validation Tool")
    print("=" * 50)
    
    # Create validator
    validator = MappingValidator(mapping_file)
    
    # Run validation
    success = validator.run_all_checks()
    
    # Save report
    validator.save_report("validation_report.json")
    
    # Show result
    if success:
        print("\n🎉 Validation completed! Your mapping file is correct.")
    else:
        print("\n⚠️ Validation completed, but some issues were found. Please check the detailed report above.")
    
    return success

if __name__ == "__main__":
    main() 