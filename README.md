# genomics
Using Runtime Reconfiguration to Accelerate   
                          Short Read Alignment







Student Name: Lubna Ibrahim Hamed Ibrahim Habeb.

Doctor Name: Sara Metwally.

Faculty Name: Computers and Information sciences.

University Name: Mansoura University. 



Abstract : Recent advances in the throughput of next-generation sequencing machines have created a tremendous computational challenge in analysing massive amounts of sequenced data. fabricated Short is an important first step in genomic data analysis. The research workflow is slowed by read alignment, which is a bottleneck. The use of a reconfigurable architecture is investigated in this article. to speed up the process Bowtie2's seed-and-extend model was used. The proposed method integrates all available information from sequencing data into an FPGA align-ment pipeline for biologically accurate runtime acceleration. Experiments show that when compared to Bowtie2, our architecture achieves a similar alignment rate while mapping reads twice as quickly. The alignment time is decreased in particular from 50 minutes to 26 minutes when processing 300M reads.

Introduction : NGS machines are capable of generating millions of short DNA fragments from the sampled cells within hours. These fragments are produced by randomly segmenting the sampled DNA strand.The orientation and location information of the reads with respect to the original DNA is lost as a result of this action. As a result, a critical first move of Quick read alignment (mapping) is a form of genomic analysis.The NGS machine generates short reads, which are mapped to a genome that serves as a guide.For the past decade, however, NGS technology has progressed much faster than Moore's Law. Because of the time it takes to sync reads with machines, it becomes unworkable.[2] time for execution This makes it difficult to use cannabis for medicinal purposes.Prenatal diagnostics and tracking are two examples of NGS applications [3] لا, [4].where the DNA and RNA of individuals should be easily analysed for an affordable price As a result, it's important to speed things up.where the DNA and RNA of individuals can be examined easily and cheaply. As a result, short read alignment must be accelerated in order to close the distance between alignment research and implementation.as well as training.Because of its highly parallel bit-wised architecture, FPGA has shown to be a promising candidate for accelerating short read mapping [5]. As the alphabet is generated in a specific order by the NGS devices can be broken down into the letters A, C, G, T, and N. 3 bits are used to represent the DNA nucleotides.to the reference genome is a bitwise operation by definition.Different algorithms have been implemented and accelerated on FPGA, such as the FM-index [6] and Smith-Waterman [7], because they are commonly used in popular software such as Bowtie2 [8] and BWA-MEM [9] are two examples.Despite its success, FPGA-accelerated alignment is still a work in progress.In genomic research and clinical applications, it is rarely used.There are two reasons for this: • The majority of accelerators have failed to make full use of the information available in NGS data. For the sake of hardware simplicity,They only use the Watson-Crick alphabets A, G, C, and T.ignoring the quality metric data and ambiguous characters (N characters) that are frequently found in NGS information.This can lead to incorrect alignment and biologically invalid results, which can be disastrous for applications like prenatal diagnostics and monitoring.Many FPGA researchers choose and accelerate alignment algorithms that favour hardware implementations. As a result, the alignment workflow may be inconsistent with cutting-edge software. For example, Arram et al. [10] propose an FPGA aligner that performs exact string matching (ESM) based on the FM-index, followed by approximate string matching (ASM) based on the seed-and-extend model. Only unmapped reads are routed to the ASM for processing in order to improve hardware performance. This methodology, on the other hand, is incompatible with the alignment model.BWA [11], Bowtie [12], and others are examples of popular software.Bowtie2, limiting biological validity and Reproducibility of hardware-generated alignment outputs.To improve the utility of short read mapping accelerators in real-world applications, we propose a two-stage alignment architecture similar to the seed-and-extend model. Bowtie2 We decouple the seeding and extension stages into two separate FPGA configurations using the runtime reconfigurability of FPGA to achieve a balanced,The alignment pipeline has been completely optimised. Finally, we take into account complete NGS data in its entirety, including quality metrics (Phred quality) in additionduring the alignment, ambiguous characters (N characters) toruntime to ensure precision.The following are the main contributions of this work: • A novel alignment architecture consisting of a two-stage configuration alignment pipeline.It takes advantage of FPGA reconfigurability to achieve highly efficient implementation for each configuration as well as a fully optimised alignment pipeline.A hardware implementation of the reconfigurable architecture aimed at a single FPGA. The seeding stage is based on FM-index implementation, and the extension stage is based on a Smith-Waterman implementation with affine-gap model.An assessment of the optimal hardware architecture based on the target platform Xilinx VU9P, with comparisons to the cutting-edge software Bowtie2 on multi-core processors and some existing FPGA solutions.

Related work : i(x) is the number of characters in R that are lexicographically smaller than x, while c(n, x) is the number of occurrences of x in BW T(R) from position 0 to position n 1. Table Ia shows how to calculate the BWT of an example reference genome R = CACGT. The suffix array is formed by the strings preceding the ‘$' sign in the sorted rotation list (SA),which indicates where each possible suffix appears in the original reference The i(x) and c(n, x) functions for the reference R are shown in Table Ib.The FM-index is used to align reads recursively on the functions i(x) and c(n, x). To perform the search, two pointers are defined at the top and bottom. The top index of the SA element is where a specific pattern can be found first, and the bottom index is where the pattern can be found last. If bottom indicates an index that is less than or equal to the index indicated by top,the pattern does not occur on the text.To search for a specific pattern P with the FM-index, one proprietarycharacter is processed at a time, starting with the last characterof P.  The top and bottom are first. first referenced with the first and last indices of the c(n, x),topnew = c(topcurrent, x) + i(x)

bottomnew = c(bottomcurrent, x) + i(x)

TABLE I: (a) Example of deriving the suffix array and BWT
of reference genome R. (b) i(x) and c(n, x) functions for R.
(a)
R = CACGT$
Index n SA Sorted Rotations
0 5 $CACGT
1 1 ACGT$C
2 0 CACGT$
3 2 CGT$CA
4 3 GT$CAC
5 4 T$CACG
BW T(R) = TC$ACG
(b)
c(n, x)
Index n A C G T
0 0 0 0 0
1 0 0 0 1
2 0 1 0 1
3 0 1 0 1
4 1 1 0 1
5 1 2 0 1
6 1 2 1 1
i(x) {1, 2, 4, 5}

