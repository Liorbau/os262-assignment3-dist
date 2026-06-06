#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  exit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return fork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

uint64
sys_sleep(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// sys_flip_display: zero-copy page flip.
//
// Syscall argument 0: user virtual address of a page-aligned buffer
// that is exactly GPU_FB_PAGES (300) * PGSIZE bytes (i.e. 640x480x4 =
// 1,228,800 bytes).  The buffer must already be fully mapped in the
// calling process's address space.
uint64
sys_flip_display(void)
{
  uint64 buf;
  argaddr(0, &buf);                 // arg 0 = user VA of the page-aligned buffer

  struct proc *p = myproc();

  // Must be page-aligned (the device backs whole pages).
  if (buf % PGSIZE != 0)
    return -1;

  // Validate: every one of the 300 pages must be mapped with user perm.
  // walkaddr returns 0 if the page is missing OR lacks PTE_U.
  for (int i = 0; i < GPU_FB_PAGES; i++) {
    if (walkaddr(p->pagetable, buf + (uint64)i * PGSIZE) == 0)
      return -1;
  }

  // Re-point the device at this process's buffer (zero-copy).
  virtio_gpu_flip(p->pagetable, buf);

  p->flip_va = buf;                   // remember which buffer was flipped
  p->flipped = 1;                   // remember, for safe cleanup on exit (Step 3)
  return 0;
}

// sys_map_display: map the GPU's kernel framebuffer pages (fb[]) directly
// into the calling process's address space with PTE_U|PTE_R|PTE_W.
//
// Syscall argument 0: desired user virtual address (must be page-aligned).
//   Pass 0 to let the kernel auto-select the next available VA above p->sz.
//
// Returns the mapped virtual address on success, (uint64)-1 on failure.
uint64
sys_map_display(void)
{
  uint64 addr;
  argaddr(0, &addr);                 // read syscall arg 0 (the requested VA)

  struct proc *p = myproc();
  uint64 size = GPU_FB_PAGES * PGSIZE;

  // --- Decide the virtual address to map at ---
  uint64 va;
  if (addr == 0) {
    // Auto-select from a high user VA region below TRAPFRAME so future
    // heap growth (sbrk) does not collide with this mapping.
    va = PGROUNDDOWN(TRAPFRAME - size);
  } else {
    // Caller-supplied VA: must be page-aligned.
    if (addr % PGSIZE != 0)
      return -1;
    va = addr;
  }

  // Sanity: the whole region must fit below MAXVA and not wrap around.
  if (va + size > MAXVA || va + size < va)
    return -1;

  // --- Collision check: none of the target pages may already be mapped ---
  for (uint64 a = va; a < va + size; a += PGSIZE) {
    pte_t *pte = walk(p->pagetable, a, 0);   // alloc=0: just look, don't create
    if (pte != 0 && (*pte & PTE_V))
      return -1;                              // something is already here
  }

  // --- Install the mapping, one kernel fb page at a time ---
  for (int i = 0; i < GPU_FB_PAGES; i++) {
    uint64 pa = gpu_fb_pa(i);
    if (pa == 0) {
      // Should never happen, but roll back cleanly if it does.
      uvmunmap(p->pagetable, va, i, 0);       // do_free = 0: kernel owns pages
      return -1;
    }
    if (mappages(p->pagetable, va + (uint64)i * PGSIZE, PGSIZE, pa,
                 PTE_U | PTE_R | PTE_W) != 0) {
      uvmunmap(p->pagetable, va, i, 0);       // undo the i pages mapped so far
      return -1;
    }
  }

  // Remember it so we can take it down when the process exits.
  p->fb_va = va;

  return va;                                  // success: hand the VA to userspace
}
