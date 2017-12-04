#include "mainwindow.h"
#include <QApplication>
#include <QDialog>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QDialog dialog;
    dialog.setWindowTitle("Hello, dialog!");
    dialog.exec();


    return a.exec();
}
