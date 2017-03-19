CXX = icpc
CXXFLAGS =

OBJECTS = matrix.o

.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

all: runme

runme: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o runme $(OBJECTS)

run-cpu: runme
	./runme

queue-cpu: runme
	echo "OFFLOAD_REPORT=2 OFFLOAD_DEVICES=none `pwd`/runme" | \
		qsub -l nodes=1:knc:coprocessor -N lab-203-cpu

queue-mic: runme
	echo "OFFLOAD_REPORT=2 `pwd`/runme" | qsub -l nodes=1:knc:coprocessor -N lab-203-cpu

clean: 
	rm -f *.o runme lab-203-cpu.o* lab-203-cpu.e*

