linuxgcc:
	g++ main.cpp  Tuple.cpp boosted_classify.cpp  -lboost_thread -lpthread -O4 -o rtrank
mac:
	g++ main.cpp  dtnode.cpp dataset.cpp  boostedtrees.cpp -I/opt/local/include/ -L/opt/local/lib -lboost_thread-mt -O4 -o rtrank -lboost_system-mt
intel:
	icc main.cpp  Tuple.cpp boosted_classify.cpp  -lboost_thread -lpthread -static -fast -o rtrank
