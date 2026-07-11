/* linker.ld — bare-metal linker script for dgemm_riscv on gem5 RISC-V M-mode
 *
 * Memory map (matching existing binary layout observed from dgemm_riscv.dis):
 *
 *   0x00010000  FLASH/ROM  — code + rodata            (existing binary base)
 *   0x00020490  RAM start  — data, BSS                (just above existing BSS end 0x20488)
 *   0x00020490  heap start — grows upward
 *   0x00120490  stack top  — grows downward (1 MB stack)
 *
 * gem5 RiscvBareMetal ELF loader maps segments at their linked addresses,
 * so RAM must start at 0x0 in gem5 config (or use --mem-start=0x0).
 * Alternatively, relink at 0x80000000; see note at bottom.
 *
 * HTIF tohost/fromhost must be 64-byte aligned for gem5 to detect them.
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    /* Code + read-only data — loaded by gem5 ELF loader.
     * 4 MB: the original 128 KB was sized for hello_world_test alone and
     * overflowed by ~700 KB linking tflm_benchmark (pulls in the full
     * kernel library plus a compiled-in .tflite model as a C array in
     * .rodata) — bumped generously so other TFLM binaries fit too,
     * without needing to retune this per binary. gem5's FS config sets
     * system.mem_ranges = [AddrRange("512MB")] starting at 0, so FLASH+RAM
     * together (8 MB total below) have ample room underneath that. */
    FLASH (rx)  : ORIGIN = 0x00010000, LENGTH = 0x00400000   /* 4 MB */

    /* Read-write data, BSS, heap, and stack. Also bumped to 4 MB (from
     * 2 MB) for the same reason — larger benchmark tensor arenas
     * (static uint8_t tensor_arena[...] in .bss) need headroom too. */
    RAM   (rw)  : ORIGIN = 0x00410000, LENGTH = 0x00400000   /* 4 MB */
}

/*
 * Stack configuration
 *   _STACK_SIZE : size of the stack in bytes (1 MB default)
 *   Grows downward from _stack_top toward _stack_bottom.
 */
_STACK_SIZE = 0x100000;   /* 1 MB */

/*
 * Heap configuration
 *   Grows upward from _heap_start toward _heap_end.
 *   _HEAP_SIZE : maximum heap size (512 KB default; tune for your workload)
 */
_HEAP_SIZE  = 0x080000;   /* 512 KB */

/* Explicit PHDRS: without this, GNU ld computes each PT_LOAD segment's
 * flags as the union of every input section's SHF_* flags it contains.
 * .init_array/.fini_array/.preinit_array are emitted SHF_WRITE by gcc
 * even though they're pure ROM data here (no runtime relocation of them
 * ever happens in this bare-metal build) — sharing FLASH's PT_LOAD with
 * .text then unions in that W bit, producing a FLASH segment marked RWE
 * instead of RE. -Wl,--fatal-warnings (as used by TFLM's Makefile) turns
 * that "LOAD segment with RWX permissions" linker warning into a build
 * failure. Forcing flags explicitly here sidesteps the union entirely. */
PHDRS
{
    flash PT_LOAD FLAGS(5); /* R+X, no W */
    ram   PT_LOAD FLAGS(6); /* R+W, no X */
}

SECTIONS
{
    /* ------------------------------------------------------------------ */
    /* .text.init MUST come first so _start lands at the ELF entry point.  */
    /* ------------------------------------------------------------------ */
    .text.init :
    {
        KEEP(*(.text.init))
    } > FLASH :flash

    /* ------------------------------------------------------------------ */
    /* Remaining code                                                       */
    /* ------------------------------------------------------------------ */
    .text :
    {
        *(.text)
        *(.text.*)
        *(.gnu.linkonce.t.*)
    } > FLASH :flash

    /* ------------------------------------------------------------------ */
    /* Read-only data                                                       */
    /* ------------------------------------------------------------------ */
    .rodata :
    {
        *(.rodata)
        *(.rodata.*)
        *(.gnu.linkonce.r.*)
    } > FLASH :flash

    /* ------------------------------------------------------------------ */
    /* HTIF communication region — must be 64-byte aligned.                */
    /* gem5 scans the ELF symbol table for "tohost" and "fromhost".        */
    /* ------------------------------------------------------------------ */
    .tohost :
    {
        . = ALIGN(64);
        KEEP(*(.tohost))
    } > FLASH :flash

    /* ------------------------------------------------------------------ */
    /* C++ init / fini arrays                                               */
    /* ------------------------------------------------------------------ */
    .preinit_array :
    {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } > FLASH :flash

    .init_array :
    {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);
    } > FLASH :flash

    .fini_array :
    {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } > FLASH :flash

    /* ------------------------------------------------------------------ */
    /* Initialized data — loaded from FLASH, copied to RAM at startup.     */
    /* (start.S does NOT copy .data; add a copy loop if you use globals.)  */
    /* ------------------------------------------------------------------ */
    .data :
    {
        . = ALIGN(8);
        _data_start = .;
        *(.data)
        *(.data.*)
        *(.gnu.linkonce.d.*)

        /* RISC-V small-data: without these, the linker orphans any
         * *(.sdata*) input section (gcc's gp-relative optimization for
         * small globals — this is where newlib puts _impure_ptr,
         * __malloc_*, __lock_* etc.) into its own PROGBITS section
         * placed right after .data but OUTSIDE [_data_start,_data_end),
         * so start.S's copy loop below never initializes it: it silently
         * reads back as zero at runtime instead of its real FLASH value,
         * which nulls out _impure_ptr and breaks all buffered stdio
         * (printf/puts silently produce no output, no crash) without
         * ever touching .text/.rodata/ordinary .data. */
        *(.sdata)
        *(.sdata.*)
        *(.gnu.linkonce.s.*)

        /* Global pointer anchor — must be here for linker relaxation. */
        . = ALIGN(8);
        PROVIDE(__global_pointer$ = . + 0x800);

        _data_end = .;
    } > RAM AT > FLASH :ram

    _data_load = LOADADDR(.data);   /* source address in FLASH */

    /* ------------------------------------------------------------------ */
    /* BSS — zero-initialized; start.S zeroes this region at boot.         */
    /* (NOLOAD), same reasoning as .heap/.stack below: no real ELF file    */
    /* content is needed since start.S's own zeroing loop (using the      */
    /* _bss_start/_bss_end VMA symbols below) handles initialization at   */
    /* runtime, not the loader. Without NOLOAD, ld tries to give .bss a   */
    /* real LMA continuing from .data's AT>FLASH address within the same  */
    /* :ram segment, which for a large .bss (e.g. tflm_benchmark's        */
    /* 150 KB+ tensor_arena) produced                                     */
    /* "section `.bss' can't be allocated in segment 1" — the same class  */
    /* of LMA-tracking confusion that .heap/.stack hit first.             */
    /* ------------------------------------------------------------------ */
    .bss (NOLOAD) :
    {
        . = ALIGN(8);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.gnu.linkonce.b.*)
        *(COMMON)

        /* RISC-V small-bss counterpart to .sdata above (see comment
         * there) — errno, __lock_* mutexes, __stdio init flag, malloc
         * bookkeeping all live here. Must be inside [_bss_start,_bss_end)
         * so start.S's zeroing loop actually covers them. */
        *(.sbss)
        *(.sbss.*)
        *(.gnu.linkonce.sb.*)
        *(.scommon)

        . = ALIGN(8);
        _bss_end = .;

        /* newlib reent symbols — not exported by all newlib builds */
        PROVIDE(_global_impure_ptr = 0x1fba0);
        PROVIDE(_impure_ptr        = 0x1fba0);
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Heap — grows upward from _heap_start.                               */
    /* (NOLOAD): pure runtime reservation, no ELF file content at all —    */
    /* without it, the linker tries to fold this into the same :ram        */
    /* PT_LOAD segment as .data and compute a contiguous LMA for it        */
    /* (chasing .data's AT>FLASH load address across half a megabyte of    */
    /* space that was never meant to be "loaded" from anywhere), which     */
    /* produces a bogus/adjusted-LMA linker warning — fatal under          */
    /* -Wl,--fatal-warnings.                                                */
    /* ------------------------------------------------------------------ */
    .heap (NOLOAD) :
    {
        . = ALIGN(16);
        _heap_start = .;
        . += _HEAP_SIZE;
        _heap_end = .;
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Stack — grows downward from _stack_top.                             */
    /* _stack_top is loaded into sp as the very first instruction.         */
    /* (NOLOAD): see .heap above.                                          */
    /* ------------------------------------------------------------------ */
    .stack (NOLOAD) :
    {
        . = ALIGN(16);
        _stack_bottom = .;
        . += _STACK_SIZE;
        . = ALIGN(16);
        _stack_top = .;             /* sp is initialized to this address */
    } > RAM

    /* ------------------------------------------------------------------ */
    /* Discard unwanted sections                                            */
    /* ------------------------------------------------------------------ */
    /DISCARD/ :
    {
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
        *(.comment)
        *(.debug*)
    }
}

/*
 * NOTE — relinking at 0x80000000 (gem5 default reset target):
 *
 * If you want gem5 to boot directly to _start without a boot ROM,
 * change FLASH ORIGIN to 0x80000000 and update gem5 mem config:
 *
 *   system.mem_ranges = [AddrRange("2GB")]
 *
 * Then compile with:
 *   riscv64-unknown-elf-gcc ... -T linker.ld -Wl,--no-relax ...
 *
 * The current layout (0x10000 base) matches the existing dgemm_riscv binary.
 */
