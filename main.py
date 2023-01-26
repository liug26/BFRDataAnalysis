from PyQt5.QtWidgets import QLabel, QListWidget, QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, \
    QCheckBox, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys
import data
import pandas as pd

# a list of color chars, can change this to add more colors
colors = list("bgrcmy")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)


class GraphWindow(QWidget):
    def __init__(self, size_width=-1, size_height=-1):
        super().__init__()
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.sc, self)
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.sc)
        self.setLayout(layout)
        if size_width == -1 or size_height == -1:
            self.setFixedSize(layout.sizeHint())
        else:
            self.setFixedSize(size_width, size_height)

    def graph(self, data_types, params):
        # sort data into groups that share the same units, so that they can share the same y-axis
        data_groups = []

        if len(data_types) < 1:
            print("No data selected")
            return

        while len(data_types) > 0:
            units = data_types[0].units
            group = []
            i = 0
            while i < len(data_types):
                if data_types[i].units == units:
                    print("Adding " + data_types[i].title + " to group")
                    group.append(data_types[i])
                    data_types.remove(data_types[i])
                else:
                    i += 1
            data_groups.append(group)
        print("Grouping finished")

        # fig = self.sc.figure
        ax = self.sc.axes
        ax.set_xlabel("Time (s)")
        lines = []
        color_index = 0
        concat_title = data_groups[0][0].title
        ax.set_ylabel(data_groups[0][0].title + " (" + data_groups[0][0].units + ")")
        for d in data_groups[0]:
            line = ax.plot(d.x, d.y, "", label=d.title, color=colors[color_index % len(colors)])
            lines += line  # note that line is a list itself
            color_index += 1
        if params["grid1"]:
            ax.grid()

        if len(data_groups) > 1:
            axis = ax.twinx()
            for d in data_groups[1]:
                line = axis.plot(d.x, d.y, "", label=d.title, color=colors[color_index % len(colors)])
                lines += line  # note that line is a list itself
                color_index += 1
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(data_groups[1][0].title + " (" + data_groups[1][0].units + ")")
            concat_title += " and " + data_groups[1][0].title
            if params["grid2"]:
                axis.grid()

        if len(data_groups) > 2:
            print("More than two units found, ignoring")

        # user can overwrite these stylistic elements
        if params["legend"]:
            ax.legend(lines, [line.get_label() for line in lines])
        if params["graph title"] != "":
            ax.set_title(params["graph title"])
        else:
            ax.set_title(concat_title + " vs Time")
        self.sc.figure.tight_layout()


def init_label_text(label_str, layout):
    panel = QWidget()
    label = QLabel(label_str, panel)
    label.setFont(QFont("Arial", 15))
    label.setGeometry(0, 7, 100, 15)
    line_edit = QLineEdit(panel)
    line_edit.setFont(QFont("Arial", 15))
    line_edit.setGeometry(110, 5, 280, 20)
    layout.addWidget(panel)
    return line_edit


def show_message_box(message, window_title, message_type):
    msg = QMessageBox()
    msg.setWindowTitle(window_title)
    msg.setIcon(message_type)
    msg.setText(message)
    msg.setFont(QFont("Arial", 15))
    msg.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # set up main window
        main_panel = QWidget()
        layout = QVBoxLayout()
        main_panel.setLayout(layout)
        self.setCentralWidget(main_panel)
        self.setWindowTitle("BFR Graphing Tool")
        self.resize(420, 680)

        # 1st row: an image and a title
        panel1 = QWidget()
        panel1.setFixedSize(420, 50)
        self.image_label = QLabel(panel1)
        pixmap = QPixmap("cat.jpg")
        self.image_label.setPixmap(pixmap)
        self.image_label.setGeometry(90, 0, 100, 50)
        title_label = QLabel("BFR Graphing Tool", panel1)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setGeometry(110, 0, 300, 50)
        title_label.setFont(QFont("Arial", 20))
        layout.addWidget(panel1)

        # 2nd row: select file
        panel2 = QWidget()
        select_file_label = QLabel("Select file", panel2)
        select_file_label.setFont(QFont("Arial", 15))
        select_file_label.setGeometry(0, 7, 100, 15)
        self.select_file_button = QPushButton("No file selected", panel2)
        self.select_file_button.setGeometry(100, 0, 300, 30)
        self.select_file_button.setFont(QFont("Arial", 15))
        self.select_file_button.clicked.connect(self._on_select_file_clicked)
        layout.addWidget(panel2)

        # 3rd row: select what to graph
        panel3 = QWidget()
        panel3.setFixedSize(420, 385)
        select_data_label = QLabel("Select data", panel3)
        select_data_label.setFont(QFont("Arial", 15))
        select_data_label.setGeometry(0, 7, 100, 15)
        self.select_data_list = QListWidget(panel3)
        self.select_data_list.addItems(data_type + " [No data]" for data_type in data.all_data)
        self.select_data_list.setSelectionMode(QListWidget.MultiSelection)
        self.select_data_list.setFont(QFont("Arial", 15))
        self.select_data_list.setGeometry(110, 0, 280, 385)
        layout.addWidget(panel3)

        # 4th row: title
        self.graph_title_text = init_label_text("Graph title", layout)
        # 5th row: graph width
        self.graph_width_text = init_label_text("Graph width", layout)
        # 6th row: graph height
        self.graph_height_text = init_label_text("Graph height", layout)

        # row 7: legend, grid, multi axis check boxes
        panel7 = QWidget()
        self.legend_check = QCheckBox("Legend", panel7)
        self.legend_check.setFont(QFont("Arial", 15))
        self.legend_check.setGeometry(80, 0, 100, 30)
        self.grid1_check = QCheckBox("Grid1", panel7)
        self.grid1_check.setFont(QFont("Arial", 15))
        self.grid1_check.setGeometry(180, 0, 100, 30)
        self.grid2_check = QCheckBox("Grid2", panel7)
        self.grid2_check.setFont(QFont("Arial", 15))
        self.grid2_check.setGeometry(270, 0, 100, 30)
        layout.addWidget(panel7)

        # row 8: graph button
        panel8 = QWidget()
        panel8.setFixedSize(420, 30)
        graph_button = QPushButton("Graph", panel8)
        graph_button.setGeometry(80, 0, 120, 30)
        graph_button.setFont(QFont("Arial", 15))
        graph_button.clicked.connect(self._on_graph_clicked)
        save_csv_button = QPushButton("Save CSV", panel8)
        save_csv_button.setGeometry(220, 0, 120, 30)
        save_csv_button.setFont(QFont("Arial", 15))
        save_csv_button.clicked.connect(self._on_save_csv_clicked)
        layout.addWidget(panel8)

    def _on_select_file_clicked(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV files (*.csv *.CSV)")
        if file_dialog.exec_():
            csv_path = file_dialog.selectedFiles()[0]
            data.load(csv_path)
            self.select_file_button.setText(csv_path.split("/")[-1])
            self.select_data_list.clear()
            self.select_data_list.addItems(data_type + " [No data]" if len(data.select_choices([data_type])[0].x) == 0
                                           else data_type for data_type in data.all_data)
            show_message_box("File imported", "Success", QMessageBox.NoIcon)

    def _on_graph_clicked(self):
        data_selected = [item.text().replace(" [No data]", "") for item in self.select_data_list.selectedItems()]
        if len(data_selected) == 0:
            show_message_box("Select at least one data to graph", "Warning", QMessageBox.Warning)
            return
        params = {
            "graph title": self.graph_title_text.text(),
            "legend": self.legend_check.isChecked(),
            "grid1": self.grid1_check.isChecked(),
            "grid2": self.grid2_check.isChecked()
        }
        data_types = data.select_choices(data_selected)
        if self.graph_width_text.text() != "" and self.graph_width_text.text().isnumeric() and \
                self.graph_height_text.text() != "" and self.graph_height_text.text().isnumeric():
            self.graph_window = GraphWindow(int(self.graph_width_text.text()), int(self.graph_height_text.text()))
        else:
            self.graph_window = GraphWindow()
        self.graph_window.graph(data_types, params)
        self.graph_window.show()

    def _on_save_csv_clicked(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec_():
            csv_dir = file_dialog.selectedFiles()[0]
            data_selected = [item.text().replace(" [No data]", "") for item in self.select_data_list.selectedItems()]
            data.save_data1(csv_dir + "/data.csv", data_selected)
            show_message_box("CSV saved", "Success", QMessageBox.NoIcon)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == '__main__':
    main()
