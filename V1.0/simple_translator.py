#!/usr/bin/env python3
"""
Zero Dharma Cultivation Project - Simple Image Translator
My first working translator!
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pytesseract
from deep_translator import GoogleTranslator
import os

class SimpleTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Zero Dharma Translator")
        self.root.geometry("800x600")
        
        # Initialize translator
        pass  # We'll create translator when needed
        self.current_image_path = None
        
        # Build simple UI
        self.setup_ui()
        
    def setup_ui(self):
        """Create the simplest possible UI"""
        
        # Title
        title = tk.Label(
            self.root, 
            text="🌏 Zero Dharma Translator", 
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)
        
        # Big button to load image
        self.load_button = tk.Button(
            self.root,
            text="📁 Click to Load Image",
            font=("Arial", 16),
            bg="#4CAF50",
            fg="white",
            width=25,
            height=2,
            command=self.load_image
        )
        self.load_button.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to translate!",
            font=("Arial", 12),
            fg="gray"
        )
        self.status_label.pack(pady=5)
        
        # Text output area
        tk.Label(self.root, text="Translation Results:", font=("Arial", 14, "bold")).pack(pady=(20, 5))
        
        self.text_output = scrolledtext.ScrolledText(
            self.root,
            width=70,
            height=15,
            font=("Arial", 12),
            wrap=tk.WORD
        )
        self.text_output.pack(pady=10, padx=20)
        
    def load_image(self):
        """Load and process image"""
        
        # Select file
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.status_label.config(text="Processing image...", fg="blue")
        self.root.update()
        
        try:
            # Extract text with OCR
            extracted_text = pytesseract.image_to_string(file_path)
            
            if not extracted_text.strip():
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(1.0, "No text found in image. Try another image.")
                self.status_label.config(text="No text detected", fg="red")
                return
                
            # Translate text
            translator = GoogleTranslator(source='auto', target='en')
            translation = translator.translate(extracted_text)

            # Show results
            output = f"ORIGINAL TEXT:\n{'-'*50}\n{extracted_text}\n\n"
            output += f"TRANSLATED TO ENGLISH:\n{'-'*50}\n{translation}"
            
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(1.0, output)
            
            self.status_label.config(text="✅ Translation complete!", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {e}")
            self.status_label.config(text="Error occurred", fg="red")

def main():
    root = tk.Tk()
    app = SimpleTranslator(root)
    root.mainloop()

if __name__ == "__main__":
    main()