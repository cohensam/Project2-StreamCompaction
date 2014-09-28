### Questions 
1) Compare this version to the serial version of exclusive prefix scan. Please
  include a table of how the runtimes compare on different lengths of arrays.
  
2) Plot a graph of the comparison and write a short explanation of the phenomenon you
  see here.
  
3) Compare this version to the parallel prefix sum using global memory.

4) Plot a graph of the comparison and write a short explanation of the phenomenon
  you see here.
  
5) Compare your version of stream compact to your version using thrust.  How do
  they compare?  How might you optimize yours more, or how might thrust's stream
  compact be optimized.
  
  3 4 6    7 9 10    11 15 20
  0 3 7    0 7 16    0  11 26
  
  7 16 26
  0 7 23
  
  0 3 7    7 
  
