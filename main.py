from fastapi import FastAPI , File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
from groq import Groq
from typing import List , Dict
import json
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from AI response that may be wrapped in markdown code blocks"""
    # Remove markdown code block markers
    cleaned_text = response_text.strip()
    
    # Remove ```json and ``` markers
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text[7:]  # Remove ```json
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text[3:]   # Remove ```
    
    if cleaned_text.endswith('```'):
        cleaned_text = cleaned_text[:-3]  # Remove closing ```
    
    # Fix backtick-wrapped strings that should be JSON strings
    # Replace backtick-wrapped multiline strings with proper JSON strings
    import re
    
    # Pattern to match: "field": `content` where content can span multiple lines
    backtick_pattern = r'("(?:code|description|wiring_description|[^"]*)":\s*)`([^`]*)`'
    
    def replace_backticks(match):
        field_name = match.group(1)
        content = match.group(2)
        # Escape content for JSON: escape quotes and newlines
        escaped_content = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return f'{field_name}: "{escaped_content}"'
    
    cleaned_text = re.sub(backtick_pattern, replace_backticks, cleaned_text, flags=re.DOTALL)
    
    return cleaned_text.strip()

# Load environment variables from .env file
load_dotenv()

app =  FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate GROQ API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "your_actual_api_key_here":
    logger.error("GROQ_API_KEY is not set or is using the placeholder value. Please set your actual API key in the .env file.")
    raise ValueError("GROQ_API_KEY is not properly configured")

groq_client = Groq(api_key=api_key)

# Simple in-memory cache for schematics
schematic_cache = {}

class Component(BaseModel):
    name:str
    type:str
    confidence:float

class Project(BaseModel):
    title:str
    description:str
    difficulty:str
    components_needed:List[str]
    code:str
    wiring_description:str


class ProjectResponse(BaseModel):
    components:List[Component]
    projects:List[Project]

# New models for schematic generation
class Position(BaseModel):
    x: float
    y: float

class Connection(BaseModel):
    from_component: str
    from_pin: str
    to_component: str
    to_pin: str

class SchematicComponent(BaseModel):
    name: str
    type: str
    position: Position
    pins: List[str]

class Circuit(BaseModel):
    components: List[SchematicComponent]
    connections: List[Connection]
    svg_data: str  # SVG markup for the schematic

@app.post("/identify-components",response_model=ProjectResponse)
async def identify_components(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file upload: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        base64_image= base64.b64encode(image_data).decode('utf-8')
        logger.info("Image successfully encoded to base64")

        component_prompt = """
        Analyze this image of electronic components and identify each component visible.
        Return ONLY a valid JSON object with this structure. Use double quotes for all strings:
        {
            "components": [
                {
                    "name": "component_name",
                    "type": "component_type",
                    "confidence": 0.95
                }
            ]
        }
        Common components to look for: Arduino boards, breadboard, LEDs, resistors, sensors, jumper wires, buttons, potentiometers, motors, displays.
        
        IMPORTANT: Return valid JSON only, no markdown formatting, use double quotes for all strings.
        """

        try:
            logger.info("Calling GROQ API for component identification with vision model")
            completion = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # Vision-capable model
                messages=[
                    {
                        "role":"user",
                        "content":[
                            {"type":"text","text": component_prompt},
                            {
                                "type":"image_url",
                                "image_url":{
                                    "url":f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            logger.info("GROQ API call successful")
            
            # Parse component response
            component_text = completion.choices[0].message.content
            logger.info(f"Raw component response: {component_text[:300]}...")
            
            try:
                # Extract JSON from markdown-wrapped response
                cleaned_json = extract_json_from_response(component_text)
                logger.info(f"Cleaned JSON: {cleaned_json[:200]}...")
                
                component_data = json.loads(cleaned_json)
                components = [Component(**comp) for comp in component_data.get("components", [])]
                
                if not components:
                    raise ValueError("No components found in AI response")
                    
                logger.info(f"Successfully parsed {len(components)} components")
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse component JSON: {str(e)}")
                logger.error(f"Original response: {component_text}")
                raise HTTPException(status_code=422, detail=f"AI returned invalid component data: {str(e)}")
            except Exception as e:
                logger.error(f"Error creating Component objects: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process component data")
                
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"GROQ API call failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Vision AI service unavailable: {str(e)}")
        
        # Generate project suggestions
        component_names = [comp.name for comp in components]
        logger.info(f"Generating projects for components: {component_names}")
        projects = await generate_projects(component_names)
        
        logger.info("Successfully completed component identification and project generation")
        return ProjectResponse(components=components, projects=projects)
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Unexpected error in identify_components: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

async def generate_projects(components: List[str]) -> List[Project]:
    """Generate project ideas based on available components"""
    
    project_prompt = f"""
    Based on these available electronic components: {', '.join(components)}
    
    Generate 3 simple Arduino projects that can be built with these components.
    Return ONLY a valid JSON object with this structure. IMPORTANT: Use double quotes for ALL string values, including code blocks:
    {{
        "projects": [
            {{
                "title": "Project Name",
                "description": "Brief description of what the project does",
                "difficulty": "Beginner/Intermediate/Advanced",
                "components_needed": ["list", "of", "components"],
                "code": "Complete Arduino code here with proper escaping",
                "wiring_description": "Step by step wiring instructions"
            }}
        ]
    }}
    
    CRITICAL: 
    - Use double quotes (") for all strings, never backticks (`)
    - Escape newlines in code as \\n
    - Escape double quotes in strings as \\"
    - Make sure the Arduino code is complete and functional
    - Return valid JSON only, no markdown formatting
    """
    
    try:
        logger.info("Generating project suggestions with GROQ API")
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated to use an available model
            messages=[{"role": "user", "content": project_prompt}],
            max_tokens=3000
        )
        
        project_text = completion.choices[0].message.content
        logger.info(f"Raw project response: {project_text[:300]}...")
        
        try:
            # Extract JSON from markdown-wrapped response
            cleaned_json = extract_json_from_response(project_text)
            logger.info(f"Cleaned project JSON: {cleaned_json[:200]}...")
            
            project_data = json.loads(cleaned_json)
            projects = [Project(**proj) for proj in project_data.get("projects", [])]
            
            if not projects:
                raise ValueError("No projects found in AI response")
                
            logger.info(f"Successfully parsed {len(projects)} projects")
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse project JSON: {str(e)}")
            logger.error(f"Original response: {project_text}")
            raise ValueError(f"AI returned invalid project data: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating Project objects: {str(e)}")
            raise ValueError(f"Failed to process project data: {str(e)}")
        
    except Exception as e:
        logger.error(f"Project generation failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Project generation AI service failed: {str(e)}")
    
    return projects

@app.post("/generate-schematic", response_model=Circuit)
async def generate_schematic(components: List[str]):
    """Generate schematic data for given components"""
    try:
        # Create cache key from sorted components
        cache_key = tuple(sorted(components))

        # Check cache
        if cache_key in schematic_cache:
            logger.info("Returning cached schematic")
            return schematic_cache[cache_key]

        logger.info(f"Generating schematic for components: {components}")

        schematic_prompt = f"""
        Generate a professional circuit schematic layout for these electronic components: {', '.join(components)}

        Return ONLY a valid JSON object with this structure. Use double quotes for all strings:
        {{
            "components": [
                {{
                    "name": "component_name",
                    "type": "component_type",
                    "position": {{"x": 100.0, "y": 200.0}},
                    "pins": ["pin1", "pin2", "GND", "VCC"]
                }}
            ],
            "connections": [
                {{
                    "from_component": "Arduino",
                    "from_pin": "D13",
                    "to_component": "LED",
                    "to_pin": "anode"
                }}
            ],
            "svg_data": "<svg width='600' height='400'><!-- Professional schematic SVG with proper symbols --></svg>"
        }}

        IMPORTANT:
        - Create professional electronic symbols (resistors as zigzag lines, LEDs as diode symbols, etc.)
        - Use proper schematic layout with Arduino on the left, components arranged logically
        - Include pin labels and connection dots
        - Use standard schematic conventions (power rails at top/bottom)
        - Generate clean, readable SVG with proper spacing
        - Return valid JSON only, no markdown formatting

        ELECTRONIC SYMBOLS TO USE:
        - Arduino: Rectangle with rounded corners, label inside
        - LED: Triangle pointing right with line at base, circle at tip
        - Resistor: Zigzag line (3-4 zigs)
        - Button: Two terminals with switch symbol
        - Breadboard: Rectangle with holes indicated
        - Wires: Straight lines with connection dots
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": schematic_prompt}],
            max_tokens=2000,
            temperature=0.3
        )

        schematic_text = completion.choices[0].message.content
        logger.info(f"Raw schematic response: {schematic_text[:300]}...")

        try:
            cleaned_json = extract_json_from_response(schematic_text)
            schematic_data = json.loads(cleaned_json)

            circuit = Circuit(**schematic_data)

            # Cache the result
            schematic_cache[cache_key] = circuit

            logger.info("Successfully generated and cached schematic")
            return circuit

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse schematic JSON: {str(e)}")
            logger.error(f"Original response: {schematic_text}")
            raise HTTPException(status_code=422, detail=f"AI returned invalid schematic data: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schematic generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schematic generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Arduino Project Cookbook API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
