# 理论手册

## 1. 控制方程

### 1.1 N-S方程的形式

我们先简单阐述一下N-S方程的推导:考虑一个方形控制体$\delta x 
\times \delta y$，这个方形控制体一共有四个边，我们按照习惯将其定义为N,S,E,W（东南西北），假设在中心点定义了物理量$\phi(x,y,z,t) = \phi (\boldsymbol{r},t)$，那么我们可以推出

$$ \phi_N = \phi + \frac{1}{2} \frac{\partial \phi}{\partial x} \delta x $$

同理可以得到其他边界。由于我们要求气体的质量不可发生改变，那么就有一个公认的关系：内部+外部=0，也就是说：

$$ \frac{\partial}{\partial t}(\rho \delta x\delta y \delta z) + \phi_N +\phi_S+\phi_E+\phi_W = \delta V(\frac{\partial \rho}{\partial t} + \nabla \cdot \rho) = 0 $$

这样就得到了以下方程组：

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \boldsymbol{u}) = 0$$

$$\frac{\partial (\rho \boldsymbol{u})}{\partial t} + \nabla \cdot (\rho \boldsymbol{u \otimes \boldsymbol{u}}) = -\nabla p \cdot \boldsymbol{I} + \nabla \cdot \boldsymbol{\tau}$$

$$\frac{\partial (\rho e)}{\partial t} + \nabla \cdot (\rho \boldsymbol{u} e ) = - \nabla \cdot (p \boldsymbol{u}) +\nabla \cdot (\lambda \nabla T) + (\boldsymbol{\tau} : \nabla \boldsymbol{u}) $$

其中$\boldsymbol{\tau} : \nabla \boldsymbol{u}$意为双点积，表示为$\boldsymbol{a} : \boldsymbol{b} = \sum \sum a_{ij}b_{ij} = \mathrm{tr} (\boldsymbol{a}\boldsymbol{b}^\top)$

### 1.2 Guass

以动量方程为例，我们先变形：

$$\frac{\partial (\rho \boldsymbol{u})}{\partial t} = - \nabla \cdot (\rho \boldsymbol{u \otimes \boldsymbol{u}})  -\nabla p \cdot \boldsymbol{I} + \nabla \cdot \boldsymbol{\tau}$$

两边对控制体$\Omega$积分，不难得到

$$\int_\Omega \frac{\partial (\rho \boldsymbol{u})}{\partial t}\mathrm{d} V = \int_\Omega \left (- \nabla \cdot (\rho \boldsymbol{u \otimes \boldsymbol{u}}) -\nabla p \cdot \boldsymbol{I} + \nabla \cdot \boldsymbol{\tau}\right ) \mathrm{d} V $$

其中右侧进行Guass：

$$ \mathrm{RHS} =- \int_{\partial \Omega}  (\rho \boldsymbol{u \otimes \boldsymbol{u}} + p \cdot \boldsymbol{I} )  \mathrm{d} S  + \int_{\partial \Omega} \boldsymbol{\tau} \mathrm{d} S $$

其中第一项称为**对流项**，简记为$\boldsymbol{F_f}$，第二项称为**扩散项**，简记为$\boldsymbol{F_d}$

### 1.3 总的原理

假设网格中心点可以代表整个格子的物理状态，面中心点可以代表整个面的状态，(事实上，这个假设在$\delta x \times \delta y \rightarrow 0$时是必然成立的.)，方程将变为：

$$ \frac{\partial (\rho \boldsymbol{u})}{\partial t} V = \sum  (\boldsymbol{F_f}+\boldsymbol{F_d} ) S $$

那么对于所有方程都作上面的处理，尽管右侧的形式不完全一致，但是还是大体一样的。右侧的变量只和力学量显相关，在不考虑时间推进的情形下，方程变为了ode，变成了可被显式Runge-Kutta的好形式：

$$ \frac{\mathrm{d} \boldsymbol{U}}{\mathrm{d} t} = \frac{\sum (\boldsymbol{F_f}+\boldsymbol{F_d}+\boldsymbol{D}+\boldsymbol{S})S}{V} $$

其中，$\boldsymbol{U}$被称为守恒量，$\boldsymbol{D}$被称为人工粘性，$\boldsymbol{S}$被称为源项

## 2. Jameson-Schmidt-Turkel 格式

### 2.1 中心差分

不难看出，上面指出的守恒量是定义在单元中心的，但是对流、扩散、耗散、源项都是定义在面上的，如何求得面上的物理量是一个难点。

> 我们可以从一种简单的思路来看，假使*信息*从某个方向开始传播，那么在面上取这个*信息*作为自己求解的依据，是一种很合理的思路，这就是迎风格式(*Upwind*)，那么这就需要辨识信息传播的方向（对扩散项而言，没有十分明确的信息概念，未必一定迎风）。当然，迎风格式是目前比较主流的一种离散方法，有关它的阐述可以在之后详细讲。

我们这里采用了一种中心差分格式，不区分信息的传播方向，只简单的将面上的物理量由两边的网格的信息平均得到，进行平均也有多种方案，假设在任何一点满足$\boldsymbol{F_f} = \boldsymbol{\varphi(U)}$，那么

$$\boldsymbol{\frac{F_{f1}+F_{f2}}{2}} \approx \boldsymbol{\varphi\left (\frac{U_1+U_2}{2}\right )} \approx \frac{\boldsymbol{\varphi({U_1})+\varphi({U_2})}}{2}$$

都是合理的，其精度为2阶，甚至你还可以将物理量直接平均到面上，然后进行计算（不基于守恒量）...如果你尝试将多种平均方法进行再（加权）平均，就构造出了更高阶的形式。

> 关于这个为何是2阶，可以基于Taylor展开，这里就不阐述了

### 2.2 时间推进

为了介绍Runge-Kutta方法，我们先考察以下线性ode

$$ \frac{\mathrm{d} u}{\mathrm{d} t} = \lambda u $$

如果进行Euler，给定一个初值$u^{(0)}$，那么$u^{(1)} = u^{(0)} + \lambda \Delta t u^{(0)} := (1+z)u^{(0)}$，考虑Taylor展开，$e^{z} = 1+z+\frac{1}{2} z^2+\cdots $，因此Euler是1阶精度的。如何构造更高阶精度的呢？答案是在指定的迭代循环进行松弛，也就是说不完全的Euler，让$u^{(i)} = (1+\alpha_i z)u^{(i-1)}$，假设我们有5步迭代，累积下来的循环是：

$$g(z)=1 + \alpha_5 z + \alpha_5\alpha_4 z^2 +\alpha_5\alpha_4\alpha_3 z^3 +\alpha_5\alpha_4\alpha_3\alpha_2 z^4 +\alpha_5\alpha_4\alpha_3\alpha_2\alpha_1 z^5$$

代入

$$e^z = 1+z+\frac{1}{2} z^2 +\frac{1}{6}z^3 + \frac{1}{24} z^4 + \frac{1}{120} z^5 +\cdots$$

解得标准值为$\alpha_k = \frac{1}{m-k+1}$，然而在JST格式中使用的RK标准值为$ \alpha_1 = \frac{1}{4},\alpha_2 = \frac{1}{6},\alpha_3 = \frac{3}{8},\alpha_4 = \frac{1}{2},\alpha_5=1 $，这削减了RK的精度，因为只能匹配上第2阶，但是这样的修改并非是为了自降精度，而是出于稳定性考虑。

关于稳定性的问题可以参考激波管的流动那一篇文章。这里简单的用一下思想，在修改系数后，对于纯对流方程，其解基于Fourier变换的放大因子是

$$g(\mathrm{i}\theta) = 1 + \mathrm{i}\theta - \frac{1}{2}\theta^2 - \mathrm{i}\left(\frac{3}{16}\right)\theta^3 + \left(\frac{1}{32}\right)\theta^4 + \mathrm{i}\left(\frac{1}{128}\right)\theta^5+o(\theta^5)$$

这个放大因子的失稳方程$|g(\mathrm{i}\theta)| = 1$的解为$\theta=4$，然而如果不改变系数，只能允许$\theta \approx 3.3$，这个$\theta$就是表示当地时间步长的一种参数***CFL***，允许更大的CFL意味着允许更大的当地时间步长，也就意味着更少的迭代步数。

对于二维问题，CFL不是很好定义，我们考虑以下几种情况，一是速度不能穿越过多网格：$\frac{u \Delta t}{\Delta x} \le CFL$，二是声波不能穿越过多网格：$\frac{c \Delta t}{\Delta x} \le CFL$，当然这里的$\Delta x$要做一些处理，至少能够表征波在某个方向上的网格长度。对于速度问题，我们做如下处理：

$$|u \cdot \Delta t \cdot \frac{\bar{S_x}}{V}|+|v \cdot \Delta t \cdot \frac{\bar{S_y}}{V}| \le CFL$$

对于声波问题，做如下处理：

$$|c \cdot \Delta t \cdot \frac{\bar{S_y}+\bar{S_x}}{V}| \le CFL$$

继续进行放缩，将二者左边相加使之小于CFL，这样就得到了高度保险的当地时间步长。当地时间步长在稳态求解中可以作为RANS的伪时间推进步长，而在URANS时必须使用全局最小时间步长。

## 3. 边界条件

### 3.1 压力远场

压力远场已在这篇文章有过详细的解释。总的来说，**特征线朝哪边，信息取哪边的值**。熵$S$和切向速度$u_\tau$的特征线是速度$u_n$，另外两个黎曼不变量$u_n \pm \frac{2a}{\gamma-1}$的特征线是$u_n\pm a$，当流动亚声速时，后两个特征线方向相反，而流动处于超声速时，二者方向相同。有了这个观点，接下来就可以判断信息取内还是取外了，当流体净流入时，前两个特征线由外向内传播，流体净流出时，前两个特征线由内向外传播。

### 3.2 无滑移壁面

边界条件可以简单的认为是速度为0($u=v=0$)，绝热壁面($\frac{\partial T}{\partial n}=0$)，压强法向梯度为0($\frac{\partial p}{\partial n}=0$)

## 4. 人工粘性

我们先给出这个$\boldsymbol{D}$

$$\boldsymbol{D} = \lambda (\epsilon^{(2)}(U_{i,j}-U_{i,j-1})-\epsilon^{(4)}(U_{i,j+1}-3U_{i,j}+3U_{i,j-1}-U_{i,j-2}))$$

其中$\lambda = \frac{CFL}{2} \left(\frac{V_1}{\Delta t_1}+\frac{V_2}{\Delta t_2}\right)$，表示当地通量的Jacobi谱半径的一种近似，事实上在无粘流动中，谱半径就是$|u|+a$，倘使这是在无粘假设下，我们求出了CFL=1时候的允许特征速度，将其加权也就形成了同等意义下的谱半径。$\epsilon^{(2)} = \frac{1}{2}\max(\mu)$，$\epsilon^{(4)} = \max(0,k_4-\epsilon^{(2)})$，其中$k_4 = \frac{1}{128}$或$\frac{1}{64}$

$\mu = \frac{|p_{i,j+1}-2p_{i,j}+p_{i,j-1}|}{p_{i,j+1}+2p_{i,j}+p_{i,j-1}}$为激波感知因子。通过对四周面上的激波感知因子进行最大值，即可确认激波存在的方位，更重要的，当三个压强趋同时，$\mu \approx 0$，而激波在本方位最为剧烈时，$\mu\rightarrow 1$，此时$\boldsymbol{D} = \frac{\lambda}{2}(U_{i,j}-U_{i,j-1})$，恰好是迎风格式的附加项，JST格式通过人工粘性完成了格式的切换。

综上，JST人工粘性的数值原理是利用压力传感器作为开关，动态调节中心差分格式的截断误差阶数：在光滑流场中引入4阶粘性压制高频振荡，在激波间断处激活2阶粘性使格式退化为迎风，从而获得锐利的激波捕捉效果。

## 5. 湍流模型

不难看出，上面的N-S方程有诸多变量是不封闭的，因此需要引入湍流模型进行封闭。**Spalart-Allmaras (S-A) 湍流模型**是由Philippe R. Spalart和Steven R. Allmaras于1992年提出的。它是一个单方程涡粘性模型，其核心是求解一个与湍流涡粘性相关的变量 $\tilde{\nu}$（称为“工作变量”或“改进湍流粘度”,在Fluent中被称为`nut`）的输运方程。

S-A模型不直接求解湍流涡粘系数 $\nu_t$，而是通过一个输运方程求解工作变量 $\tilde{\nu}$。湍流涡粘系数 $\nu_t$ 则由 $\tilde{\nu}$ 通过一个阻尼函数 $f_{v1}$ 计算得出：

$$
\nu_t = \tilde{\nu} f_{v1}, \quad f_{v1} = \frac{\chi^3}{\chi^3 + C_{v1}^3}, \quad \chi \equiv \frac{\tilde{\nu}}{\nu}
$$

其中，$\nu$ 是分子运动粘度，$C_{v1} = 7.1$ 是一个模型常数。在近壁面处，$\tilde{\nu}$ 与壁面距离呈线性关系，这使得模型对近壁面网格分辨率不那么敏感，鲁棒性更好。S-A模型的核心是 $\tilde{\nu}$ 的输运方程。其非守恒形式可写为：

$$
\frac{\partial \tilde{\nu}}{\partial t} + u_j \frac{\partial \tilde{\nu}}{\partial x_j} = 
\underbrace{C_{b1} [1 - f_{t2}] \tilde{S} \tilde{\nu}}_{\text{生成项 } P}\\ 
+ \underbrace{\frac{1}{\sigma} \left\{ \nabla \cdot [(\nu + \tilde{\nu}) \nabla \tilde{\nu}] + C_{b2} (\nabla \tilde{\nu})^2 \right\}}_{\text{扩散项 } D}\\ 
- \underbrace{\left[ C_{w1} f_w - \frac{C_{b1}}{\kappa^2} f_{t2} \right] \left( \frac{\tilde{\nu}}{d} \right)^2}_{\text{破坏（耗散）项 } \varepsilon}
+ \underbrace{f_{t1} \Delta U^2}_{\text{转捩项}}
$$

方程左边为时间项和对流项，右边为扩散项和由转捩项、生成项、破坏项构成的**源项**，基于上面各个项分立的思想，可以把这个粘度的输运方程也写进守恒量、对流项、扩散项、源项中。这个湍流模型还有大量的内部处理，且形式各异，这里不多赘述，具体可参阅原始论文。求解完毕`nut`后，可以以此来封闭Reynold应力$\tau_{xx},\tau_{xy},\tau_{yy}$和导热系数$\lambda_{\mathrm{eff}}$

## 6. 流程

在完成网格读取、初始化后，即可进入求解过程，我们先建立守恒量$\boldsymbol{U} = (\rho,\rho u,\rho v,\rho e,\rho \tilde{\nu})^\top$，以此形成控制方程的左边。然后根据Guass定理建立面上的对流项、扩散项、人工粘性和源项，以此合成控制方程的右边。最后使用RK进行显式时间迭代，即完成1次外循环。