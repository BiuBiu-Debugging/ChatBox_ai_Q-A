import tkinter as tk
from tkinter import font as tkfont
import threading
import datetime
import time

import faiss

from Lmd_Utils import chatbot_TF_IDF,Chatbot_Rag_embed
import joblib

#vect=joblib.load('./models_TF-IDF/tfidf_QA_VN.pkl')
#X_train=joblib.load('./models_TF-IDF/Question_QA_VN.pkl')
#answers=joblib.load('./models_TF-IDF/Answer_QA_VN.pkl')
#questions=joblib.load( './models_TF-IDF/Question_QA_VN.pkl')

index = faiss.read_index('./models_rag/faiss_index.bin')
answers=joblib.load('./models_rag/answers.pkl')
questions=joblib.load( './models_rag/questions.pkl')


C = {
    "bg":          "#09090E",
    "panel":       "#0F1017",
    "sidebar":     "#0C0C13",
    "input_bg":    "#13141C",
    "input_focus": "#181924",
    "border":      "#1E2030",
    "border_hi":   "#2E3255",
    "user_bubble": "#0F2040",
    "ai_bubble":   "#111320",
    "accent":      "#4F8EF7",
    "accent2":     "#7C6FFF",
    "glow":        "#2A4A8A",
    "success":     "#3DD68C",
    "error":       "#FF6B6B",
    "warn":        "#FFB347",
    "txt_primary": "#E2E4F0",
    "txt_secondary":"#7B82A0",
    "txt_muted":   "#3E4460",
    "txt_user":    "#BDD5FF",
    "txt_ai":      "#DDE1F5",
    "scrollbar":   "#1A1C2E",
}

MONO = "Courier New"
SANS = "Trebuchet MS"



class ChatboxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Chatbox")
        self.geometry("900x680")
        self.minsize(640, 480)
        self.configure(bg=C["bg"])
        self._setup_fonts()
        self._build_ui()
        self.conversation: list[dict] = []
        self._typing_job = None
        self._is_thinking = False
        self.after(100, self.input_box.focus_set)
        self.index=index
        self.answers=answers
        self.questions=questions


    def _setup_fonts(self):
        self.f_title = tkfont.Font(family=MONO, size=13, weight="bold")
        self.f_body  = tkfont.Font(family=MONO, size=10)
        self.f_bold  = tkfont.Font(family=MONO, size=10, weight="bold")
        self.f_small = tkfont.Font(family=MONO, size=8)
        self.f_meta  = tkfont.Font(family=SANS, size=8)
        self.f_input = tkfont.Font(family=MONO, size=11)
        self.f_btn   = tkfont.Font(family=MONO, size=10, weight="bold")
        self.f_label = tkfont.Font(family=SANS, size=9, weight="bold")

    def _build_ui(self):
        self._build_titlebar()
        self._build_body()
        self._build_statusbar()

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=C["sidebar"], height=48)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=C["accent"], width=3).pack(side="left", fill="y")
        logo_frame = tk.Frame(bar, bg=C["sidebar"])
        logo_frame.pack(side="left", padx=(14, 0), pady=8)
        tk.Label(logo_frame, text="◈", font=self.f_title,
                 fg=C["accent"], bg=C["sidebar"]).pack(side="left", padx=(0, 8))
        tk.Label(logo_frame, text="AI CHATBOX", font=self.f_title,
                 fg=C["txt_primary"], bg=C["sidebar"]).pack(side="left")
        tk.Label(logo_frame, text=" v1.0", font=self.f_small,
                 fg=C["txt_muted"], bg=C["sidebar"]).pack(side="left", pady=(4, 0))
        right = tk.Frame(bar, bg=C["sidebar"])
        right.pack(side="right", padx=16)
        tk.Label(right, text="●", font=self.f_small,
                 fg=C["success"], bg=C["sidebar"]).pack(side="left", padx=(0, 4))
        tk.Label(right, text="LOCAL MODEL", font=self.f_label,
                 fg=C["success"], bg=C["sidebar"]).pack(side="left")

    def _build_body(self):
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        chat_frame = tk.Frame(body, bg=C["panel"],
                              highlightbackground=C["border"],
                              highlightthickness=1)
        chat_frame.pack(fill="both", expand=True, padx=12, pady=(10, 0))

        self.canvas = tk.Canvas(chat_frame, bg=C["panel"],
                                highlightthickness=0, cursor="arrow")
        sb = tk.Scrollbar(chat_frame, orient="vertical",
                          command=self.canvas.yview,
                          bg=C["scrollbar"], troughcolor=C["bg"],
                          activebackground=C["border_hi"])
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.msg_frame = tk.Frame(self.canvas, bg=C["panel"])
        self.canvas_win = self.canvas.create_window(
            (0, 0), window=self.msg_frame, anchor="nw")

        self.msg_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>",   self._on_mousewheel)
        self.canvas.bind_all("<Button-5>",   self._on_mousewheel)

        self._show_welcome()

        self.think_frame = tk.Frame(body, bg=C["bg"], height=28)
        self.think_frame.pack(fill="x", padx=12)
        self.think_frame.pack_propagate(False)
        self.think_label = tk.Label(self.think_frame, text="",
                                    font=self.f_meta, fg=C["accent"],
                                    bg=C["bg"])
        self.think_label.pack(side="left", padx=8, pady=4)

        input_outer = tk.Frame(body, bg=C["border"], pady=1)
        input_outer.pack(fill="x", padx=12, pady=(0, 12))
        input_inner = tk.Frame(input_outer, bg=C["input_bg"])
        input_inner.pack(fill="x")

        tk.Label(input_inner, text=" ❯ ", font=self.f_bold,
                 fg=C["accent"], bg=C["input_bg"]).pack(side="left")

        self.input_var = tk.StringVar()
        self.input_box = tk.Entry(input_inner, textvariable=self.input_var,
                                  font=self.f_input, fg=C["txt_primary"],
                                  bg=C["input_bg"], insertbackground=C["accent"],
                                  relief="flat", bd=0)
        self.input_box.pack(side="left", fill="both", expand=True, ipady=10)
        self.input_box.bind("<Return>", self._on_send)
        self.input_box.bind("<FocusIn>",
            lambda e: input_outer.config(bg=C["accent"]))
        self.input_box.bind("<FocusOut>",
            lambda e: input_outer.config(bg=C["border"]))

        self.clear_btn = tk.Button(input_inner, text="CLR", font=self.f_btn,
                                   fg=C["txt_secondary"], bg=C["input_bg"],
                                   activebackground=C["input_focus"],
                                   activeforeground=C["txt_primary"],
                                   relief="flat", bd=0, padx=8, cursor="hand2",
                                   command=self._clear_chat)
        self.clear_btn.pack(side="right", padx=(0, 4), pady=4)
        _hover(self.clear_btn, C["input_bg"], C["input_focus"],
               C["txt_secondary"], C["txt_primary"])

        self.send_btn = tk.Button(input_inner, text="SEND ▶", font=self.f_btn,
                                  fg=C["bg"], bg=C["accent"],
                                  activebackground=C["accent2"],
                                  activeforeground=C["bg"],
                                  relief="flat", bd=0, padx=14, cursor="hand2",
                                  command=self._on_send)
        self.send_btn.pack(side="right", padx=(0, 2), pady=4)
        _hover(self.send_btn, C["accent"], C["accent2"], C["bg"], C["bg"])

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["sidebar"], height=22)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self.status_var, font=self.f_small,
                 fg=C["txt_muted"], bg=C["sidebar"]).pack(side="left", padx=10)
        self.msg_count_var = tk.StringVar(value="0 messages")
        tk.Label(bar, textvariable=self.msg_count_var, font=self.f_small,
                 fg=C["txt_muted"], bg=C["sidebar"]).pack(side="right", padx=10)

    def _show_welcome(self):
        pad = tk.Frame(self.msg_frame, bg=C["panel"])
        pad.pack(fill="x", pady=(24, 0))
        for text, fnt, color in [
            ("◈  AI CHATBOX", self.f_title, C["accent"]),
            ("", self.f_small, C["panel"]),
            ("Type a message and press Enter or click SEND.", self.f_body, C["txt_secondary"]),
            ("", self.f_small, C["panel"]),
        ]:
            tk.Label(pad, text=text, font=fnt, fg=color, bg=C["panel"]).pack()
        tk.Frame(self.msg_frame, bg=C["border"], height=1).pack(
            fill="x", padx=24, pady=8)

    def _add_message(self, role: str, text: str):
        is_user = (role == "user")
        outer = tk.Frame(self.msg_frame, bg=C["panel"])
        outer.pack(fill="x", padx=16, pady=(6, 0))

        header = tk.Frame(outer, bg=C["panel"])
        header.pack(fill="x")
        icon_char  = "▶ YOU" if is_user else "◈ AI"
        icon_color = C["accent"] if is_user else C["accent2"]
        tk.Label(header, text=icon_char, font=self.f_bold,
                 fg=icon_color, bg=C["panel"]).pack(side="left")
        tk.Label(header, text=f"  {datetime.datetime.now().strftime('%H:%M')}",
                 font=self.f_meta, fg=C["txt_muted"], bg=C["panel"]).pack(side="left")

        bubble_bg = C["user_bubble"] if is_user else C["ai_bubble"]
        txt_color  = C["txt_user"]   if is_user else C["txt_ai"]
        bubble_bd  = C["glow"]       if is_user else C["border"]

        bubble = tk.Frame(outer, bg=bubble_bg,
                          highlightbackground=bubble_bd, highlightthickness=1)
        bubble.pack(fill="x", pady=(4, 2))

        msg_label = tk.Label(bubble, text=text, font=self.f_body,
                             fg=txt_color, bg=bubble_bg,
                             wraplength=720, justify="left", anchor="w",
                             padx=14, pady=10)
        msg_label.pack(fill="x")
        bubble.bind("<Configure>",
            lambda e, lbl=msg_label: lbl.config(wraplength=max(200, e.width - 32)))

        self._scroll_bottom()
        self._update_count()

    def _add_error(self, text: str):
        outer = tk.Frame(self.msg_frame, bg=C["panel"])
        outer.pack(fill="x", padx=16, pady=4)
        tk.Label(outer, text=f"⚠ ERROR: {text}", font=self.f_body,
                 fg=C["error"], bg=C["panel"], anchor="w").pack(fill="x")
        self._scroll_bottom()

    def _start_thinking(self):
        self._is_thinking = True
        self.send_btn.config(state="disabled", bg=C["border"])
        self.status_var.set("AI is thinking…")
        self._animate_thinking()

    def _stop_thinking(self):
        self._is_thinking = False
        if self._typing_job:
            self.after_cancel(self._typing_job)
        self.think_label.config(text="")
        self.send_btn.config(state="normal", bg=C["accent"])
        self.status_var.set("Ready")

    def _animate_thinking(self):
        if not self._is_thinking:
            return
        dots = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        idx = int(time.time() * 8) % len(dots)
        self.think_label.config(text=f"  {dots[idx]}  generating response…")
        self._typing_job = self.after(80, self._animate_thinking)
    def _on_send(self, event=None):
        text = self.input_var.get().strip()
        if not text or self._is_thinking:
            return
        self.input_var.set("")
        self._add_message("user", text)
        self.conversation.append({"role": "user", "content": text})
        self._start_thinking()
        threading.Thread(target=self._fetch_reply, args=(text,), daemon=True).start()

    def _fetch_reply(self, text: str):
        try:
            a = Chatbot_Rag_embed(text, self.index)
            result =''
            if (len(a)>0):
                result=answers[a[0]]
            else:
                result="Xin lỗi, tôi không có dữ liệu cho câu hỏi của bạn"
            error=None
            reply=result
        except Exception as exc:
            reply = None
            error = str(exc)
        self.after(0, self._on_reply, reply, error)

    def _on_reply(self, reply, error):
        self._stop_thinking()
        if error:
            self._add_error(error)
        else:
            self._add_message("assistant", reply)
            self.conversation.append({"role": "assistant", "content": reply})

    def _clear_chat(self):
        for widget in self.msg_frame.winfo_children():
            widget.destroy()
        self.conversation.clear()
        self._show_welcome()
        self._update_count()
        self.status_var.set("Chat cleared")
        self.after(2000, lambda: self.status_var.set("Ready"))

    def _update_count(self):
        n = len(self.conversation)
        self.msg_count_var.set(f"{n} message{'s' if n != 1 else ''}")

    def _scroll_bottom(self):
        self.after(50, lambda: self.canvas.yview_moveto(1.0))

    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_win, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def _hover(widget, bg, bg_h, fg, fg_h):
    widget.bind("<Enter>", lambda e: widget.config(bg=bg_h, fg=fg_h))
    widget.bind("<Leave>", lambda e: widget.config(bg=bg,   fg=fg))


if __name__ == "__main__":
    app = ChatboxApp()
    app.mainloop()