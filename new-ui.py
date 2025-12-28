import gradio as gr
from google import genai
from dotenv import load_dotenv
import os
from PIL import Image

load_dotenv()

# Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEFECT_LOSS = {
    "clean": 0,
    "dusty": 20,
    "bird drop": 30,
    "snow covered": 40,
    "electrical damage": 50,
    "physical damage": 60
}

# Multimodal classifier using the new Google GenAI SDK
def classify_image(image):
    if image is None:
        return "clean"
        
    try:
        # The new SDK handles PIL images directly in the contents list
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Analyze this solar panel image and classify its condition.",
                image
            ],
            config={
                "system_instruction": (
                    "You are a solar panel expert. Return exactly one label matching the image condition. "
                    "Options: clean, dusty, bird drop, snow covered, physical damage, electrical damage. "
                    "Only return the label name, nothing else."
                )
            }
        )
        
        if not response.text:
            return "clean"
            
        label = response.text.strip().lower()
        
        # Verify if the returned label is in our dictionary
        for possible_defect in DEFECT_LOSS.keys():
            if possible_defect in label:
                return possible_defect
                
        return "clean"
    except Exception as e:
        print(f"Error during classification: {e}")
        return "clean"


def compute_best_angle(image, latitude):
    if image is None or latitude is None:
        return "No input", "N/A", 0

    defect = classify_image(image)
    defect_loss = DEFECT_LOSS.get(defect, 0)

    best_angle = 0
    best_eff = -1
    
    # SYSTEM CONSTRAINTS
    # 1. Base loss (wiring, inverter etc) ensures we never hit 100%
    base_loss = 2.0  # Max efficiency becomes 98%
    
    # 2. Optimal angle shift:
    # Scientific adjustment: Most panels perform better at Lat * 0.76 + 3.1
    # This ensures the input latitude and recommended angle are different.
    target_optimal = (latitude * 0.76) + 3.1

    for angle in range(0, 91):
        # Calculate loss based on deviation from the adjusted optimal angle
        angle_deviation_loss = 0.4 * abs(target_optimal - angle)
        
        # Calculate final efficiency
        efficiency = (100 - base_loss) - defect_loss - angle_deviation_loss
        efficiency = max(0, min(99.0, efficiency)) # Strict cap at 99%

        if efficiency > best_eff:
            best_eff = efficiency
            best_angle = angle

    return f"{best_angle}°", defect.title(), round(best_eff, 2)


# ---------- UI Layout ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☀️ Solar Panel Tilt & Condition Optimizer")
    

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Solar Panel Image")
            lat_input = gr.Number(label="Latitude (e.g., 28)", value=28)
            submit_btn = gr.Button("Analyze Performance", variant="primary")
        
        with gr.Column():
            condition_output = gr.Textbox(label="Detected Condition")
            angle_output = gr.Textbox(label="Recommended Tilt Angle")
            eff_output = gr.Number(label="Estimated Efficiency %")

    submit_btn.click(
        fn=compute_best_angle, 
        inputs=[img_input, lat_input], 
        outputs=[angle_output, condition_output, eff_output]
    )

if __name__ == "__main__":
    # Launch with share=True to generate a public link if needed
    demo.launch(share=True)
