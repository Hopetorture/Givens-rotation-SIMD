QT += core
QT -= gui

CONFIG += c++11

TARGET = Givens
CONFIG += console
CONFIG += -mavx2
CONFIG += -msse2
#CONFIG += -mavx
CONFIG -= app_bundle

QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp
#QMAKE_CXXFLAGS+= -mavx2
#QMAKE_LFLAGS += -mavx2
QMAKE_CXXFLAGS+= -mavx2
QMAKE_LFLAGS += -mavx2
QMAKE_CXXFLAGS+= -msse2
QMAKE_LFLAGS += -msse2


# remove possible other optimization flags
QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -Os
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

# add the desired -O3 if not present
#QMAKE_CXXFLAGS_RELEASE *= -O0
QMAKE_CXXFLAGS_RELEASE *= -O0  # O0good nice x4 increase, O1+ - godly increase

TEMPLATE = app

SOURCES += main.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
