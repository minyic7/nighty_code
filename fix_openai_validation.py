#!/usr/bin/env python3
"""
Patch script to fix OpenAI validation issue
This modifies the validate_connection method to use models.list() as fallback
"""

import os
import sys


def patch_openai_provider():
    """Apply the validation fix to OpenAI provider"""
    
    file_path = "src/llm/providers/openai.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # The original validation method
    original_validation = '''    async def validate_connection(self) -> bool:
        """Validate the connection to OpenAI"""
        try:
            # Try a minimal API call to validate the connection
            response = await self.client.models.retrieve(self.config.model)
            return response.id == self.config.model
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            return False'''
    
    # The fixed validation method
    fixed_validation = '''    async def validate_connection(self) -> bool:
        """Validate the connection to OpenAI - Fixed to use models.list() as fallback"""
        try:
            # Method 1: Try the original validation first (works for standard OpenAI)
            try:
                response = await self.client.models.retrieve(self.config.model)
                if response.id == self.config.model:
                    return True
            except Exception:
                pass  # Fallback to Method 2
            
            # Method 2: List models and check if our model exists (works for proxies/Azure)
            models_response = await self.client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if self.config.model in available_models:
                logger.info(f"Model '{self.config.model}' validated using models.list()")
                return True
            else:
                logger.warning(f"Model '{self.config.model}' not found in available models")
                return False
                
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            return False'''
    
    # Check if already patched
    if "models.list()" in content:
        print("✅ File already patched!")
        return True
    
    # Apply the patch
    if original_validation in content:
        content = content.replace(original_validation, fixed_validation)
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Successfully patched OpenAI provider!")
        print("   The validation now uses models.list() as fallback")
        return True
    else:
        print("❌ Could not find the original validation method to patch")
        print("   The file may have been modified")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI Validation Fix")
    print("=" * 60)
    print()
    
    if patch_openai_provider():
        print("\n✅ Patch applied successfully!")
        print("   Your OpenAI client should now work on the company laptop")
    else:
        print("\n❌ Patch failed")
        print("   You may need to manually edit src/llm/providers/openai.py")