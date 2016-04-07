CXX = g++
OBJ = obj
INC = include
BIN = bin
LIB = lib
SRC = src
CXXFLAGS =-Wall -g -c -O1 -m64
OPENCV = `pkg-config opencv --cflags --libs`
CVSBA = `pkg-config cvsba --cflags --libs`
OMP = -fopenmp
LIBS = $(OPENCV) $(CVSBA)

main:$(OBJ)/main.o $(OBJ)/proyecto.o
	$(CXX) -o main $^ $(LIBS) $(OMP)

$(OBJ)/main.o: $(SRC)/main.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/main.cpp -o $(OBJ)/main.o -I$(INC)

$(OBJ)/proyecto.o: $(SRC)/proyecto.cpp $(INC)/proyecto.h
	$(CXX) $(CXXFLAGS) $(SRC)/proyecto.cpp -o $(OBJ)/proyecto.o -I$(INC) $(OMP)

clean:
	 rm *~; rm $(SRC)/*~; rm $(INC)/*~; rm $(OBJ)/*;rm *.txt; rm mai*; rm *.jpg;

Gitclean:
	 git rm *~; git rm $(SRC)/*~; git rm $(INC)/*~; git rm $(OBJ)/*;
