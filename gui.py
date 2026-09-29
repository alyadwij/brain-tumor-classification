import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from preprocess import preprocess_image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from classify import predict

# Load input image
def load_input():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return
    
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

    # Preprocess and predict
    selected_model = classification_mode.get()

    if selected_model == "CNN":
        if mode.get() == "Preview Preprocessing":
            img_preprocessed, steps = preprocess_image(file_path, return_steps=True)

            if hasattr(root, "preview_window") and root.preview_window.winfo_exists():
                try:
                    root.preview_window.destroy()
                except:
                    pass
            
            root.preview_window = tk.Toplevel(root)
            preview_window = root.preview_window
            root.preview_window.title("Preprocessing Steps")

            row = 0
            col = 0

            for name, step_img in steps.items():
                img_show = Image.fromarray(step_img)
                img_show = img_show.resize((120, 120))
                img_tk = ImageTk.PhotoImage(img_show)

                label_img = tk.Label(preview_window, image=img_tk)
                label_img.image = img_tk
                label_img.grid(row=row, column=col)

                label_text = tk.Label(preview_window, text=name)
                label_text.grid(row=row+1, column=col)

                col += 1
                if col == 3:
                    col = 0
                    row += 2
        else:
            img_preprocessed = preprocess_image(file_path)
    else:
        if selected_model != "CNN" and mode.get() == "Preview Preprocessing":
            messagebox.showwarning(
                "Preview Unavailable",
                "Preprocessing preview is only available for the CNN model."
            )
            
        img = Image.open(file_path).convert("RGB")
        img = img.resize((224, 224))

        img_preprocessed = np.array(img)
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        if selected_model == "ResNet50":
            img_preprocessed = preprocess_resnet(img_preprocessed)
        elif selected_model == "VGG-16":
            img_preprocessed = preprocess_vgg16(img_preprocessed)
        elif selected_model == "VGG-19":
            img_preprocessed = preprocess_vgg19(img_preprocessed)

    label, conf, probs = predict(img_preprocessed, selected_model)
    
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    if label == "notumor":
        result_label.config(fg="green") 
        main_text = (f"Prediction: No Tumor ({conf*100:.2f}%)")
    else:
        result_label.config(fg="red") 
        main_text = (f"Prediction: {label.upper()} ({conf*100:.2f}%)")
    
    # Detail probabilitas
    prob_text = "\n\nProbabilities:\n"
    for i in range(len(classes)):
        prob_text += f"{classes[i]:<12}: {probs[0][i]*100:6.2f}%\n"
    
    result_text.set(main_text + prob_text)

def reset_app():
    panel.config(image="")
    panel.image = None

    result_text.set("")
    mode.set("Direct Classification")
    classification_mode.set("CNN")

    if hasattr(root, "preview_window") and root.preview_window:
        try:
            root.preview_window.destroy()
        except:
            pass
        root.preview_window = None

def run_app():
    global root
    root = tk.Tk()
    root.title("Brain Tumor Classification")
    root.geometry("750x500")

    # TOP FRAME
    top_frame = tk.Frame(root)
    top_frame.pack(pady=15)

    # Mode
    global mode
    mode = tk.StringVar(value="Direct Classification")

    # Classification mode
    global classification_mode
    classification_mode = tk.StringVar(value="CNN")

    # Button
    tk.Button(top_frame, text="Upload", command=load_input, bg="#002AFF", fg="white", width=10).grid(row=0, column=4, padx=10)
    tk.Button(top_frame, text="Reset", command=reset_app, bg="#FFFFFF", fg="blue", width=10).grid(row=0, column=5, padx=10)

    # Mode selection
    tk.Label(top_frame, text="Mode:").grid(row=0, column=0, padx=10)

    mode_combobox = ttk.Combobox(top_frame, textvariable=mode, values=["Direct Classification", "Preview Preprocessing"], state="readonly", width=20)
    mode_combobox.grid(row=0, column=1, padx=10)
    mode.set("Direct Classification")

    # Classification method
    tk.Label(top_frame, text="Classification Model:").grid(row=0, column=2, padx=10)

    classification_combobox = ttk.Combobox(top_frame, textvariable=classification_mode, values=["CNN", "ResNet50", "VGG-16", "VGG-19"], state="readonly", width=20)
    classification_combobox.grid(row=0, column=3, padx=10)
    classification_combobox.current(0)

    # MAIN FRAME
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    main_frame.columnconfigure(0, weight=3)
    main_frame.columnconfigure(1, weight=2)
    main_frame.rowconfigure(0, weight=1)

    # LEFT (Image)
    left_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE, padx=10, pady=10)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Title Left
    tk.Label(left_frame, text="Input Image", font=("Arial", 12, "bold")).pack(pady=5)
    
    # Divider
    tk.Frame(left_frame, height=2, bg="gray").pack(fill=tk.X, pady=5)

    # Panel left
    global panel
    panel = tk.Label(left_frame)
    panel.pack(pady=10, fill=tk.BOTH, expand=True)

    # RIGHT (Result)
    right_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE, padx=15, pady=10)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    # Title
    tk.Label(right_frame, text="Classification Result", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Divider
    tk.Frame(right_frame, height=2, bg="gray").pack(fill=tk.X, pady=5)

    # Result
    global result_text
    result_text = tk.StringVar()

    global result_label
    result_label = tk.Label(right_frame, textvariable=result_text, font=("Courier", 11), justify="left", anchor="w")
    result_label.pack(pady=10, fill=tk.BOTH)

    root.mainloop()