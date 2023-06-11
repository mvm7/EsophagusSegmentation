import tkinter as tk
import modifyImage as mI

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from threading import Thread


class windowCreator:
    def __init__(self):
        self.originalImage = Image
        self.resizedOriginalImage = Image
        self.newImage = Image
        self.resizedNewImage = Image
        self.mask = Image
        self.background = Image
        # window
        self.window = tk.Tk()
        self.window.geometry("495x500")  # размеры окна 495x500
        self.window.title("Esophagus Analyzer (VGG16 U-Net 3+)")  # заголовок окна
        self.window.resizable(False, False)  # запрет на изменение размеров окна

        # labels
        tk.Label(text="Original image", font="Arial 13").place(x=200, y=10)
        tk.Label(text="New image", font="Arial 13").place(x=710, y=10)

        # buttons
        self.btn1 = tk.Button(text="Select image", command=self.click1, font="Arial 11")
        self.btn1.place(height=30, width=460, x=20, y=385)

        self.btn2 = tk.Button(text="Segment the disease", state=tk.DISABLED, command=self.click2,
                              font="Arial 11")
        self.btn2.place(height=30, width=229, x=20, y=420)

        self.btn3 = tk.Button(text="Separate the disease", state=tk.DISABLED, command=self.click3, font="Arial 11")
        self.btn3.place(height=30, width=229, x=20 + 231, y=420)

        self.btn4 = tk.Button(text="Highlight diseases",
                              state=tk.DISABLED, command=self.click4, font="Arial 11")
        self.btn4.place(height=30, width=460, x=20, y=455)

        self.btn5 = tk.Button(text="Save new image",
                              font="Arial 12", command=self.click5)
        self.btn5.place(height=100, width=460, x=520, y=385)

        # containers
        frm1 = tk.Frame(relief=tk.RIDGE, borderwidth=2)
        frm1.place(height=340, width=460, x=20, y=40)
        frm2 = tk.Frame(relief=tk.RIDGE, borderwidth=2)
        frm2.place(height=340, width=460, x=520, y=40)

        # diseases labels
        self.label1_1 = tk.Label(text="■", font="Arial 17", foreground="#00ff00")
        self.label1_2 = tk.Label(text=" - suspicious area", font="Arial 11")
        self.label2_1 = tk.Label(text="■", font="Arial 17", foreground="#00ffff")
        self.label2_2 = tk.Label(text=" - polyps", font="Arial 11")
        self.label3_1 = tk.Label(text="■", font="Arial 17", foreground="#ff0000")
        self.label3_2 = tk.Label(text=" - high grade dysplasia", font="Arial 11")
        self.label4_1 = tk.Label(text="■", font="Arial 17", foreground="#ff00ff")
        self.label4_2 = tk.Label(text=" - Barrett's esophagus", font="Arial 11")
        self.label5_1 = tk.Label(text="■", font="Arial 17", foreground="#0000ff")
        self.label5_2 = tk.Label(text=" - esophageal cancer", font="Arial 11")

        # graphics areas
        self.canvas1 = tk.Canvas(frm1, height=340, width=460)
        self.canvas1.pack()
        self.canvas2 = tk.Canvas(frm2, height=340, width=460)
        self.canvas2.pack()

        self.window.mainloop()

    # "Select image" button
    def click1(self):
        pathToFile = tk.filedialog.askopenfilename(filetypes=(("JPG files", "*.jpg"),
                                                              ("PNG files", "*.png"),
                                                              ("JPEG files", "*.jpeg"),
                                                              ("BMP files", "*.bmp")))
        if pathToFile:
            self.originalImage = Image.open(pathToFile)
            self.resizedOriginalImage = self.originalImage
            if self.resizedOriginalImage.size[0] > 460:
                self.resizedOriginalImage = self.originalImage.resize((460, int(460 * self.originalImage.size[1]
                                                                                / self.originalImage.size[0])),
                                                                      Image.LANCZOS)
            if self.resizedOriginalImage.size[1] > 340:
                self.resizedOriginalImage = self.originalImage.resize((int(340 * self.originalImage.size[0]
                                                                           / self.originalImage.size[1]), 340),
                                                                      Image.LANCZOS)
            self.resizedOriginalImage = ImageTk.PhotoImage(self.resizedOriginalImage)
            self.canvas1.create_image(230, 170, anchor="center", image=self.resizedOriginalImage)

            self.thread1 = Thread(target=self.maskThread)
            self.thread1.start()

            self.btn2.config(state=tk.NORMAL)
            self.btn3.config(state=tk.NORMAL)
            self.btn4.config(state=tk.NORMAL)
            self.window.geometry("495x500")

    def maskThread(self):
        self.mask = mI.getMask(self.originalImage)

    # "Segment the disease" button
    def click2(self):
        self.thread1.join()
        self.newImage = self.mask
        self.resizeNewImage()
        self.canvas2.create_image(230, 170, anchor="center", image=self.resizedNewImage)

        self.btn5.place(height=40, width=460, x=520, y=445)
        self.label1_1.place(x=595, y=382)
        self.label1_2.place(x=610, y=388)
        self.label2_1.place(x=535, y=407)
        self.label2_2.place(x=550, y=413)
        self.label3_1.place(x=745, y=382)
        self.label3_2.place(x=760, y=388)
        self.label4_1.place(x=630, y=407)
        self.label4_2.place(x=645, y=413)
        self.label5_1.place(x=815, y=407)
        self.label5_2.place(x=830, y=413)
        self.window.geometry("1000x500")

    # "Separate the disease" button
    def click3(self):
        self.thread1.join()
        self.newImage = mI.deleteBackground(self.originalImage, self.mask)
        self.resizeNewImage()
        self.canvas2.create_image(230, 170, anchor="center", image=self.resizedNewImage)

        self.btn5.place(height=100, width=460, x=520, y=385)
        self.label1_1.place_forget()
        self.label1_2.place_forget()
        self.label2_1.place_forget()
        self.label2_2.place_forget()
        self.label3_1.place_forget()
        self.label3_2.place_forget()
        self.label4_1.place_forget()
        self.label4_2.place_forget()
        self.label5_1.place_forget()
        self.label5_2.place_forget()
        self.window.geometry("1000x500")

    # "Highlight diseases" button
    def click4(self):
        self.thread1.join()
        self.newImage = mI.highlights(self.originalImage, self.mask)
        self.resizeNewImage()
        self.canvas2.create_image(230, 170, anchor="center", image=self.resizedNewImage)

        self.btn5.place(height=40, width=460, x=520, y=445)
        self.label1_1.place(x=595, y=382)
        self.label1_2.place(x=610, y=388)
        self.label2_1.place(x=535, y=407)
        self.label2_2.place(x=550, y=413)
        self.label3_1.place(x=745, y=382)
        self.label3_2.place(x=760, y=388)
        self.label4_1.place(x=630, y=407)
        self.label4_2.place(x=645, y=413)
        self.label5_1.place(x=815, y=407)
        self.label5_2.place(x=830, y=413)
        self.window.geometry("1000x500")

    # "Save new image" button
    def click5(self):
        fileName = filedialog.asksaveasfilename(defaultextension="png", filetypes=(("PNG files", "*.png"),
                                                                                   ("BMP files", "*.bmp")))
        if fileName:
            self.newImage.save(fileName)
            print(5)

    # window resizing
    def resizeNewImage(self):
        self.resizedNewImage = self.newImage
        if self.resizedNewImage.size[0] > 460:
            self.resizedNewImage = self.newImage.resize((460, int(460 * self.newImage.size[1]
                                                                  / self.newImage.size[0])),
                                                        Image.LANCZOS)
        if self.resizedNewImage.size[1] > 340:
            self.resizedNewImage = self.newImage.resize((int(340 * self.newImage.size[0]
                                                             / self.newImage.size[1]), 340),
                                                        Image.LANCZOS)
        self.resizedNewImage = ImageTk.PhotoImage(self.resizedNewImage)


if __name__ == '__main__':
    mainWindow = windowCreator()
    del mainWindow
