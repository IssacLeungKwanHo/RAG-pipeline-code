import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextBrowser, QTextEdit, QPushButton, QComboBox, QLabel, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from ver2_test5_std import setup_local_chatbot

logging.basicConfig(level=logging.INFO)

class ChatWorker(QThread):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, query_function, user_input):
        super().__init__()
        self.query_function = query_function
        self.user_input = user_input

    def run(self):
        try:
            response = self.query_function(self.user_input)
            self.response_signal.emit(response)
        except Exception as e:
            self.error_signal.emit(str(e))

class ParentADHDChatbot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADHD Parent Support Chatbot - RAG")
        self.setGeometry(100, 100, 1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header = QLabel("ADHD Parenting Support Assistant (RAG Pipeline)")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Model Selection (All 9 models - same as original)
        controls = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "llama3.1:8b", "llama3.2:3b", "deepseek-r1:8b", "deepseek-r1:7b",
            "gemma3n:e4b", "gemma3:4b", "mistral:7b", "qwen3:4b", "qwen3:8b"
        ])
        self.model_combo.setCurrentText("llama3.1:8b")
        self.model_combo.currentTextChanged.connect(self.change_model)

        controls.addWidget(QLabel("AI Model:"))
        controls.addWidget(self.model_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.chat_display = QTextBrowser()
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.input_area = QTextEdit()
        self.input_area.setFixedHeight(80)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_area)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        self.current_model = "llama3.1:8b"
        self.initialize_chatbot()

    def initialize_chatbot(self):
        try:
            _, _, self.query_func, _, _ = setup_local_chatbot(self.current_model)
            self.chat_display.append("<b>System ready (Parent Mode)</b><br>")
            self.chat_display.append("How can I assist you today?<br><br>")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def change_model(self):
        self.current_model = self.model_combo.currentText()
        self.chat_display.append(f"<i>Switching to {self.current_model}...</i><br>")
        self.initialize_chatbot()

    def send_message(self):
        text = self.input_area.toPlainText().strip()
        if not text:
            return
        self.chat_display.append(f"<b>You:</b> {text}<br>")
        self.input_area.clear()

        self.worker = ChatWorker(self.query_func, text)
        self.worker.response_signal.connect(self.show_response)
        self.worker.start()

    def show_response(self, response):
        self.chat_display.append(f"<b>Assistant ({self.current_model}):</b> {response}<br><br>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParentADHDChatbot()
    window.show()
    sys.exit(app.exec_())