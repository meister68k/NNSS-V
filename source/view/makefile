.PHONY : all clean cleanall

TARGET = MNVIEW.com
all : $(TARGET)

SRCS = MNVIEW.c
OBJS = $(patsubst %.c,%.o,$(SRCS:%.z80=%.o))

AS     = zasm
ASOPT  = 
CC     = zcc +cpm -subtype=x1 -c
CCOPT  = -O3 -DAMALLOC
LK     = zcc +cpm -pragma-output:USING_amalloc -create-app
LKOPT  = 
NDC    = ../../x1/tool/ndc
DISK   = ../../x1/diskimage/data_fat12.d88


$(TARGET) : $(OBJS)
	$(LK) $(LKOPT) -o$(@:%.com=%.bin) $<

%.o : %.z80
	$(AS) $(ASOPT) $< $@ $(@:%.bin=%.lst)

%.o : %.c
	$(CC) $(CCOPT) -o$@ $<

upload : $(TARGET)
	-$(NDC) D $(DISK) 0 $(TARGET)
	$(NDC) P $(DISK) 0 $(TARGET)

clean :
	-del $(OBJS) $(TARGET:%.com=%.bin) zcc_opt.def

cleanall : clean
	-del $(TARGET)

