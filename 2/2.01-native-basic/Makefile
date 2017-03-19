hello-CPU : hello.cc
	icpc -o $@ $<

hello-KNC : hello.cc
	icpc -o $@ -mmic $<


clean:
	-rm *.o* *.e*
