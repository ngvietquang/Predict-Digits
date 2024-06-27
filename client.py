import tkinter as tk
from tkinter import Canvas, Button , Label
from PIL import Image, ImageDraw


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        self.canvas = Canvas(self.root, width=115, height=115, bg="white")
        self.canvas.pack()

        self.button_clear = Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(side=tk.LEFT, padx= 50)

        self.button_next = Button(self.root, text="Next", command=self.predict)
        self.button_next.pack(side=tk.RIGHT, padx= 50)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("RGB", (115, 115), "white")
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 3
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (115, 115), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        file_path = "picture.png"
        self.image.save(file_path)
        print("Saved Successfully")
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("Prediction")

        from Predict import NeuralNetwork

        predicted_number = NeuralNetwork()
        # predicted_number.training()
        predicted_number.load_model("./train.npz")
        
        label = Label(prediction_window, text=f"Predicted Number: {predicted_number.predict(predicted_number.conv1)}")
        
        label.pack(pady=10)
        predicted_number.show_img()

        

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
