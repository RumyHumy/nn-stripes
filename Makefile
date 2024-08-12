# gcc main.c -o Build -I/usr/include -L/usr/lib -lraylib -lm -lGL -lpthread -ldl -lrt -lX11 -std=c99

RUNN_PATH = ./runn

# Define the source files
SOURCES = main.c $(RUNN_PATH)/src/runn.c

# Define the object files
OBJECTS = $(SOURCES:.c=.o)

# Define the executable file
EXECUTABLE = Build

# Define the compiler flags
CFLAGS = -Wall -g -std=c99
CFLAGS += -I/usr/include -I$(RUNN_PATH)/include

LDFLAGS = -L/usr/lib -lraylib -lm -lGL -lpthread -ldl -lrt -lX11

# Default rule to build the executable
all: $(EXECUTABLE)

# Rule to compile the object files
%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

# Rule to link the object files into the executable
$(EXECUTABLE): $(OBJECTS)
	gcc $(OBJECTS) $(LDFLAGS) -o $@

run: all
	./Build

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
