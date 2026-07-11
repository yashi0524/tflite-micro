/* start.S — bare-metal crt0 for dgemm_riscv on gem5 RISC-V M-mode
 *
 * stdio strategy:
 *   _isatty(1) returns 1 -> newlib sets stdout LINE-BUFFERED
 *   -> printf flushes on every \n -> _write is called per line
 *   -> no fflush, no __sinit, no _global_impure_ptr needed
 *
 * _write routes output via semihosting SYS_WRITE0 (0x04).
 * _exit  terminates via semihosting SYS_EXIT  (0x18).
 *
 * The semihosting magic sequences use raw .word to prevent
 * assembler compression. They are made distinct by unique
 * padding so lld ICF cannot fold them.
 */

#define SYS_WRITE0            0x04
#define SYS_EXIT              0x18
#define ADP_STOPPED_APP_EXIT  0x20026

/* ================================================================== */
    .section .text.init, "ax", @progbits
    .globl  _start
    .type   _start, @function
_start:
    la      sp, _stack_top

    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    la      t0, trap_entry
    csrw    mtvec, t0

    /* Enable FP (mstatus.FS=Initial) and vector (mstatus.VS=Initial) */
    li      t0, 0x2200          /* FS[14:13]=01, VS[10:9]=01 */
    csrs    mstatus, t0

    /* Zero BSS */
    la      a0, _bss_start
    la      a1, _bss_end
    beq     a0, a1, .Lbss_done
.Lbss_loop:
    sd      zero, 0(a0)
    addi    a0, a0, 8
    bltu    a0, a1, .Lbss_loop
.Lbss_done:

  /* Copy .data from flash (LMA) to RAM (VMA) */                                                                
    la      t0, _data_load                                                                                        
    la      t1, _data_start                                   
    la      t2, _data_end                                                                                         
    beq     t1, t2, .Ldata_done                               
  .Ldata_loop:
    ld      t3, 0(t0)                                                                                             
    sd      t3, 0(t1)
    addi    t0, t0, 8                                                                                             
    addi    t1, t1, 8                                         
    bltu    t1, t2, .Ldata_loop
  .Ldata_done:

    /* Run C++ global/static constructors (.preinit_array + .init_array)
     * before main(). linker_semi.ld already defines
     * __{preinit,init}_array_{start,end} (needed for KEEP()'d ordering
     * regardless), but nothing was ever walking them — harmless for
     * binaries with no such constructors, but silently wrong for any
     * that register themselves this way. Concretely: TFLM's
     * testing/micro_test_v2.h TEST(...) macro (GTest-style, as opposed
     * to the older TF_LITE_MICRO_TEST macro) registers each test via a
     * `static TestRegistrar reg(...);` at namespace scope — a static
     * object with a non-trivial constructor. Without running
     * .init_array, that registration never happens: the test binary
     * still runs to completion and exits 0, but silently executes zero
     * tests ("[==========] 0 tests ran." / "~~~ALL TESTS PASSED~~~" —
     * a vacuous, not a real, pass). */
    la      s0, __preinit_array_start
    la      s1, __preinit_array_end
    j       .Lpreinit_check
  .Lpreinit_loop:
    ld      t0, 0(s0)
    jalr    t0
    addi    s0, s0, 8
  .Lpreinit_check:
    bltu    s0, s1, .Lpreinit_loop

    la      s0, __init_array_start
    la      s1, __init_array_end
    j       .Linit_check
  .Linit_loop:
    ld      t0, 0(s0)
    jalr    t0
    addi    s0, s0, 8
  .Linit_check:
    bltu    s0, s1, .Linit_loop

    li      a0, 0
    li      a1, 0
    li      a2, 0
    call    main
    /* Call libc exit() (not _exit directly): exit() flushes all open
     * stdio buffers via _fwalk before calling _exit() itself. Falling
     * through to _exit directly (as this file's upstream/gemm-project
     * copy does) loses any output that didn't happen to trigger an
     * implicit line-buffered flush — fine for gemm's kernel, which
     * always ends on a flushed newline, but not safe in general. */
    call    exit
    .size   _start, . - _start


/* ================================================================== */
/* _exit                                                               */
/* Sequence tag: 1 x nop after suffix (unique to _exit)               */
/* ================================================================== */
    .globl  _exit
    .type   _exit, @function
_exit:
    mv      s0, a0
    addi    sp, sp, -16
    li      t0, ADP_STOPPED_APP_EXIT
    sd      t0, 0(sp)
    sd      s0, 8(sp)
    li      a0, SYS_EXIT            /* 0x18 */
    mv      a1, sp
    .option push
    .option norvc
    .word   0x01f01013              /* slli x0, x0, 0x1f  PREFIX  */
    .word   0x00100073              /* ebreak              TRIGGER */
    .word   0x40705013              /* srai x0, x0, 7     SUFFIX  */
    .word   0x00000013              /* addi x0,x0,0  tag=1        */
    .option pop
    addi    sp, sp, 16
.Lhalt:
    wfi
    j       .Lhalt
    .size   _exit, . - _exit


/* ================================================================== */
/* _write                                                              */
/* Sequence tag: 2 x nop after suffix (unique to _write)              */
/* ================================================================== */
    .globl  _write
    .type   _write, @function
_write:
    beqz    a2, .Lwrite_ret

    addi    sp, sp, -48
    sd      ra, 40(sp)
    sd      s0, 32(sp)
    sd      s1, 24(sp)
    sd      s2, 16(sp)
    sd      s3,  8(sp)
    sd      s4,  0(sp)

    mv      s0, a1              /* buf */
    mv      s1, a2              /* count */

    /* Allocate null-terminated copy on stack */
    addi    t0, s1, 17
    andi    t0, t0, -16
    sub     sp, sp, t0
    mv      s2, t0
    mv      s3, sp

    mv      s4, s1
    mv      t1, s0
    mv      t2, s3
.Lwrite_copy:
    beqz    s4, .Lwrite_nul
    lbu     t3, 0(t1)
    sb      t3, 0(t2)
    addi    t1, t1, 1
    addi    t2, t2, 1
    addi    s4, s4, -1
    j       .Lwrite_copy
.Lwrite_nul:
    sb      zero, 0(t2)

    li      a0, SYS_WRITE0          /* 0x04 */
    mv      a1, s3
    .option push
    .option norvc
    .word   0x01f01013              /* slli x0, x0, 0x1f  PREFIX  */
    .word   0x00100073              /* ebreak              TRIGGER */
    .word   0x40705013              /* srai x0, x0, 7     SUFFIX  */
    .word   0x00000013              /* addi x0,x0,0  tag=1        */
    .word   0x00000013              /* addi x0,x0,0  tag=2        */
    .option pop

    add     sp, sp, s2
    ld      ra, 40(sp)
    ld      s0, 32(sp)
    ld      s1, 24(sp)
    ld      s2, 16(sp)
    ld      s3,  8(sp)
    ld      s4,  0(sp)
    addi    sp, sp, 48

    mv      a0, s1
    ret

.Lwrite_ret:
    li      a0, 0
    ret
    .size   _write, . - _write


/* ================================================================== */
/* trap_entry — M-mode trap handler                                    */
/* Dumps mcause, mepc, mtval via semihosting, then exits.             */
/* Sequence tag: 3 x nop (write), 5 x nop (exit) — unique to trap    */
/* ================================================================== */
    .section .text.init, "ax", @progbits
    .globl  trap_entry
    .type   trap_entry, @function
    .align  2
trap_entry:
    addi    sp, sp, -32
    sd      a0, 24(sp)
    sd      a1, 16(sp)
    sd      a2,  8(sp)
    sd      a3,  0(sp)

    /* fill mcause */
    csrr    a0, mcause
    la      a1, trap_mcause_hex + 15
    li      a2, 16
.Ltrap_hex0:
    andi    a3, a0, 0xf
    li      t0, 10
    blt     a3, t0, .Ltrap_d0
    addi    a3, a3, ('a' - 10)
    j       .Ltrap_s0
.Ltrap_d0: addi a3, a3, '0'
.Ltrap_s0: sb a3, 0(a1)
    addi    a1, a1, -1
    srli    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Ltrap_hex0

    /* fill mepc */
    csrr    a0, mepc
    la      a1, trap_mepc_hex + 15
    li      a2, 16
.Ltrap_hex1:
    andi    a3, a0, 0xf
    li      t0, 10
    blt     a3, t0, .Ltrap_d1
    addi    a3, a3, ('a' - 10)
    j       .Ltrap_s1
.Ltrap_d1: addi a3, a3, '0'
.Ltrap_s1: sb a3, 0(a1)
    addi    a1, a1, -1
    srli    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Ltrap_hex1

    /* fill mtval */
    csrr    a0, mtval
    la      a1, trap_mtval_hex + 15
    li      a2, 16
.Ltrap_hex2:
    andi    a3, a0, 0xf
    li      t0, 10
    blt     a3, t0, .Ltrap_d2
    addi    a3, a3, ('a' - 10)
    j       .Ltrap_s2
.Ltrap_d2: addi a3, a3, '0'
.Ltrap_s2: sb a3, 0(a1)
    addi    a1, a1, -1
    srli    a0, a0, 4
    addi    a2, a2, -1
    bnez    a2, .Ltrap_hex2

    /* SYS_WRITE0: print trap_msg (null-terminated) */
    li      a0, SYS_WRITE0
    la      a1, trap_msg
    .option push
    .option norvc
    .word   0x01f01013              /* slli x0, x0, 0x1f  PREFIX  */
    .word   0x00100073              /* ebreak              TRIGGER */
    .word   0x40705013              /* srai x0, x0, 7     SUFFIX  */
    .word   0x00000013              /* nop  tag=1                  */
    .word   0x00000013              /* nop  tag=2                  */
    .word   0x00000013              /* nop  tag=3                  */
    .option pop

    /* SYS_EXIT: terminate simulation */
    li      t0, ADP_STOPPED_APP_EXIT
    addi    sp, sp, -16
    sd      t0, 0(sp)
    sd      zero, 8(sp)
    li      a0, SYS_EXIT
    mv      a1, sp
    .option push
    .option norvc
    .word   0x01f01013              /* slli x0, x0, 0x1f  PREFIX  */
    .word   0x00100073              /* ebreak              TRIGGER */
    .word   0x40705013              /* srai x0, x0, 7     SUFFIX  */
    .word   0x00000013              /* nop  tag=1                  */
    .word   0x00000013              /* nop  tag=2                  */
    .word   0x00000013              /* nop  tag=3                  */
    .word   0x00000013              /* nop  tag=4                  */
    .word   0x00000013              /* nop  tag=5                  */
    .option pop
    addi    sp, sp, 16

.Ltrap_halt:
    wfi
    j       .Ltrap_halt
    .size   trap_entry, . - trap_entry


/* ================================================================== */
/* Newlib syscall stubs                                                */
/* ================================================================== */
    .globl  _read
    .type   _read, @function
_read:
    li      a0, -1
    ret

    .globl  _sbrk
    .type   _sbrk, @function
_sbrk:
    la      t0, _heap_ptr
    ld      t1, 0(t0)
    beqz    t1, .Lsbrk_init
    j       .Lsbrk_bump
.Lsbrk_init:
    la      t1, _heap_start
.Lsbrk_bump:
    add     t2, t1, a0
    la      t3, _heap_end
    bgtu    t2, t3, .Lsbrk_fail
    sd      t2, 0(t0)
    mv      a0, t1
    ret
.Lsbrk_fail:
    li      a0, -1
    ret

    .globl  _close
    .type   _close, @function
_close:
    li      a0, -1
    ret

    .globl  _fstat
    .type   _fstat, @function
_fstat:
    li      a0, -1
    ret

    .globl  _isatty
    .type   _isatty, @function
_isatty:
    li      a0, 1               /* return 1 = is a tty                */
    ret                         /* -> newlib sets LINE buffering mode  */
                                /* -> printf flushes on every \n      */

    .globl  _lseek
    .type   _lseek, @function
_lseek:
    li      a0, -1
    ret

    .globl  _getpid
    .type   _getpid, @function
_getpid:
    li      a0, 1
    ret

    .globl  _kill
    .type   _kill, @function
_kill:
    li      a0, 0
    j       _exit

    /* Pulled in unavoidably by TFLM's bundled eyalroz_printf printf.o
     * (whole-.o linking drags in putchar_wrapper, which references this
     * even though only vsnprintf_/printf_ — not putchar()/puts() — are
     * actually used). Routes a single byte through our own _write. */
    .globl  putchar_
    .type   putchar_, @function
putchar_:
    addi    sp, sp, -16
    sb      a0, 0(sp)
    mv      a1, sp
    li      a2, 1
    li      a0, 1
    call    _write
    addi    sp, sp, 16
    ret
    .size   putchar_, . - putchar_


/* trap message — in .data so hex fields are writable at runtime */
/* Placed after all .text.init code so the section switch doesn't  */
/* accidentally absorb the Newlib stubs into .data.                */
/* ================================================================== */
/* DATA                                                                */
/* ================================================================== */
    .section .data
    .align  3
trap_msg:
    .ascii  "\nTRAP: mcause=0x"
trap_mcause_hex:
    .ascii  "0000000000000000"
    .ascii  "  mepc=0x"
trap_mepc_hex:
    .ascii  "0000000000000000"
    .ascii  "  mtval=0x"
trap_mtval_hex:
    .ascii  "0000000000000000\n\0"


/* ================================================================== */
/* BSS                                                                 */
/* ================================================================== */
    .section .bss
    .align  3
    .globl  _heap_ptr
_heap_ptr:
    .dword  0
