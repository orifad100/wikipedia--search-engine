import tkinter
import tkinter as tk
from tkinter import messagebox
import requests
from PIL import ImageTk
from tkinter import *
from PIL import Image, ImageTk
from termcolor import colored

from tkinter import ttk
class SearchApp:
    def rgb_hack(self):
        return "#%02x%02x%02x" % (0,0,255)

    def __init__(self, master):
        self.master = master
        master.title("Wiki Search")
        master.geometry("400x500")
        master.config(bg=self.rgb_hack())
        image1 = Image.open("IMAGE.png")
        image1 = image1.resize((400,100), Image.ANTIALIAS)
        test = ImageTk.PhotoImage(image1)
        label1 = tkinter.Label(master, image=test)
        label1.image = test
        label1.pack()

        # self.query_label = tk.Label(master)
        # self.query_label.pack()
        self.query_entry = tk.Entry(master, width=20, font=('Arial 12'))
        self.query_entry.pack()

        self.search_button = tk.Button(master, bg="white", text="Search in wiki", fg='black', font=("Tahoma", 14),
                                       command=self.search)
        self.search_button.pack()

        self.results_listbox = tk.Listbox(master,height=20,width=50)
        self.results_listbox.pack()



    def search(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return

        try:
            response = requests.get(f"http://34.133.250.220:8080/search?query={query}")
            results = response.json()
            self.results_listbox.delete(0, tk.END)
            for result in results:
                self.results_listbox.insert(tk.END, result[1])
        except:
            messagebox.showerror("Error", "An error occurred. Please try again later.")

root = tk.Tk()
app = SearchApp(root)
root.mainloop()
