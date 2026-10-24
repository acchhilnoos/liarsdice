CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Iinclude

SRCDIR  := src
OBJDIR  := build
TESTDIR := tests

SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))

TARGET := main

all: CFLAGS += -O3 -mavx2 -mfma
all: $(TARGET)

debug: CFLAGS += -O0 -g -mavx2 -mfma 
debug: $(TARGET)

test: CFLAGS += -O3 -mavx2 -mfma 
test: $(OBJDIR)/game.o $(OBJDIR)/network.o $(OBJDIR)/tensor.o $(OBJDIR)/test.o
	@echo "Linking $@..."
	@$(CC) $^ -o $@ -lm

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ -lm
	@echo "Done."

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/test.o: $(TESTDIR)/test.c | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir $@

clean:
	@rm -rf $(OBJDIR) $(TARGET)
	@rm -rf *.dSYM

.PHONY: all clean debug test
