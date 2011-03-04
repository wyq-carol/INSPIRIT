PROG=exemple

.PHONY: all

all: $(PROG)


CC      := gcc
CFLAGS  := $$(pkg-config --cflags libstarpu) -Wall
LDFLAGS := $$(pkg-config --libs libstarpu)

$(PROG): %: %.o
	$(CC) -g -rdynamic -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f $(PROG)
