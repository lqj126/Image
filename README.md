# 报告内容 ：
- 成果展示
- 引擎框架概述
- 渲染框架概述
- 着色器原理概述

### 模型路径
C:/Code/Lotus/Sandbox/assets/models/nanosuit/nanosuit.obj
C:/Code/Lotus/Sandbox/assets/models/backpack/backpack.obj



# 冯光照模型
## 顶点着色器
```C++
//Material Texture Shader
#type vertex
	#version 330 core

	layout(location = 0) in vec3 a_Position;
	layout(location = 1) in vec3 a_Normal;
	layout(location = 2) in vec2 a_TexCoord;

	uniform mat4 u_ViewProjection;
	uniform mat4 u_Transform;
	uniform mat3 u_TransformNormal;

	out vec2 v_TexCoord;
	out vec3 v_Normal;
	out vec3 v_FragPosition;
```
```C++
	void main()
	{
		v_TexCoord = a_TexCoord;
        //将输入顶点的法向量变换到世界坐标系下结果赋值给输出顶点属性v_Normal
		v_Normal = u_TransformNormal * a_Normal; 
        //物体位置坐标转换到世界坐标系
		vec4 worldCoord = u_Transform * vec4(a_Position, 1.0f); 
		v_FragPosition = worldCoord.xyz;
        //将worldCoord变量与视图投影矩阵u_ViewProjection相乘，得到变换后的裁剪空间坐标
		gl_Position = u_ViewProjection * worldCoord; 
	}
```
将输入顶点的法向量变换到世界坐标系下结果赋值给输出顶点属性v_Normal  
`v_Normal = u_TransformNormal * a_Normal;`
$$
  \begin{bmatrix}
    v_{Normal,x} & v_{Normal,y} & v_{Normal,z}
  \end{bmatrix}  = 
  \begin{bmatrix}
  u_{11} & u_{12} & u_{13}\\\ u_{21} & u_{22} & u_{23}\\\ u_{31} & u_{32} & u_{33}
  \end{bmatrix} \times
  \begin{bmatrix}a_{Normal,x}\\\ a_{Normal,y}\\\ a_{Normal,z}\end{bmatrix}
\tag{1}
$$
//物体位置坐标转换到世界坐标系  
`vec4 worldCoord = u_Transform * vec4(a_Position, 1.0f);`
$$
\begin{bmatrix}
w_x\\w_y\\w_z\\w_w
\end{bmatrix} =
\begin{bmatrix}u_{11} & u_{12} & u_{13} & u_{14}\\u_{21} & u_{22} & u_{23} & u_{24}\\u_{31} & u_{32} & u_{33} & u_{34}\\u_{41} & u_{42} & u_{43} & u_{44}
\end{bmatrix} \times 
\begin{bmatrix}
a_x\\a_y\\a_z\\1
\end{bmatrix}\tag{2}
$$
//将worldCoord变量与视图投影矩阵u_ViewProjection相乘，得到变换后的裁剪空间坐标  
`gl_Position = u_ViewProjection * worldCoord;`
$$
\begin{bmatrix}
p_x\\p_y\\p_z\\p_w
\end{bmatrix} = 
\begin{bmatrix}v_{11} & v_{12} & v_{13} & v_{14}\\v_{21} & v_{22} & v_{23} & v_{24}\\v_{31} & v_{32} & v_{33} & v_{34}\\v_{41} & v_{42} & v_{43} & v_{44}
\end{bmatrix} \times 
\begin{bmatrix}w_x\\w_y\\w_z\\w_w
\end{bmatrix}\tag{3}
$$

## 片段着色器
```C++
#type fragment
	#version 330 core

	layout(location = 0) out vec4 color;

	in vec2 v_TexCoord;
	in vec3 v_Normal;
	in vec3 v_FragPosition;
	uniform vec3 u_ViewPosition;
```

```C++
//直射光
	struct DirectionalLight
	{
		vec3 direction;
		vec3 color;

		float ambient;
		float diffuse;
		float specular;
	};
	uniform DirectionalLight u_DirectionalLight;
//点光源
	struct PointLight
	{
		vec3 position;
		vec3 color;

		float ambient;
		float diffuse;
		float specular;

		float constant;
		float linear;
		float quadratic;
	};
	#define MAX_POINT_LIGHT 4
	uniform int u_PointLightCount = 0;
	uniform PointLight u_PointLights[MAX_POINT_LIGHT];
//聚光灯
	struct SpotLight
	{
		vec3 position;
		vec3 direction;
		vec3 color;

		float ambient;
		float diffuse;
		float specular;

		float constant;
		float linear;
		float quadratic;

		float cutOff;
		float outerCutOff;
	};
	uniform int u_SpotLightCount = 0;  
	uniform SpotLight u_SpotLight;
//物体材质
	struct Material
	{
		sampler2D diffuse;
		sampler2D specular;
		sampler2D emission;
		float shininess;
	};
	uniform Material u_Material;
```
```C++
//颜色采样
	vec4 diffuseColor = texture(u_Material.diffuse, v_TexCoord);
	vec4 reflectColor = texture(u_Material.specular, v_TexCoord);
    //标准化向量
	vec3 normal = normalize(v_Normal);
	vec3 viewDir = normalize(u_ViewPosition - v_FragPosition);
```
```C++
//计算直射光
	vec3 CalcDirectionalLight(DirectionalLight light)
	{
        //计算出从物体表面到光源的方向(lightDir)，用于后续的光照计算。
		vec3 lightDir = normalize(-light.direction);
	
		// 环境光
		vec3 ambient = light.ambient * light.color * diffuseColor.rgb;
	
		// 漫反射光
		float diffuseIntensity = max(dot(normal, lightDir), 0.0);
		vec3 diffuse = light.diffuse * light.color * diffuseIntensity * diffuseColor.rgb;

		// 镜面光
		vec3 reflectDir = reflect(-lightDir, normal);//reflect(I, N) = I - 2 * dot(I, N) * N
		float specularIntensity = pow(max(dot(viewDir, reflectDir), 0.0), u_Material.shininess);
		vec3 specular = light.specular * light.color * specularIntensity * reflectColor.rgb;

		return ambient + diffuse + specular;
	}
```
```C++
//计算点光源
	vec3 CalcPointLight(PointLight light)
	{
		vec3 lightDir = normalize(light.position.xyz - v_FragPosition);

		float dist= length(light.position.xyz - v_FragPosition);
		float attenuation = 1.0 / (
			light.constant + light.linear * dist +  light.quadratic * (dist * dist)
		);

		// 环境光
		vec3 ambient = light.ambient * light.color * diffuseColor.rgb;
	
		// 漫反射光
		float diffuseIntensity = max(dot(normal, lightDir), 0.0);
		vec3 diffuse = light.diffuse * light.color * diffuseIntensity * diffuseColor.rgb;

		// 镜面光
		vec3 reflectDir = reflect(-lightDir, normal);
		float specularIntensity = pow(max(dot(viewDir, reflectDir), 0.0), u_Material.shininess);
		vec3 specular = light.specular * light.color * specularIntensity * reflectColor.rgb;

		return (ambient + diffuse + specular) * attenuation;
	}
```
```C++
//计算聚光灯
	vec3 CalcSpotLight(SpotLight light)
	{
		vec3 lightDir = normalize(light.position.xyz - v_FragPosition);

		float dist= length(light.position.xyz - v_FragPosition);
		float attenuation = 1.0 / (
			light.constant + light.linear * dist +  light.quadratic * (dist * dist)
		);

		float theta = dot(lightDir, normalize(-light.direction));
		float epsilon = light.cutOff - light.outerCutOff;
		float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

		// 环境光
		vec3 ambient = light.ambient * light.color * diffuseColor.rgb;
	
		// 漫反射光
		float diffuseIntensity = max(dot(normal, lightDir), 0.0);
		vec3 diffuse = light.diffuse * light.color * diffuseIntensity * diffuseColor.rgb;

		// 镜面光
		vec3 reflectDir = reflect(-lightDir, normal);
		float specularIntensity = pow(max(dot(viewDir, reflectDir), 0.0), u_Material.shininess);
		vec3 specular = light.specular * light.color * specularIntensity * reflectColor.rgb;

		return (ambient + diffuse + specular) * attenuation * intensity;
	}
```
```C++
//后处理模式选择
	uniform int PostPorcess;
//主程序
	void main()
	{
        //初始化颜色，默认为黑色->材质提取错误时模型为黑色
		color = vec4(0.0, 0.0, 0.0, 1.0);
        //计算直射光
		vec3 directionalColor = CalcDirectionalLight(u_DirectionalLight);
		color.rgb += directionalColor;
        //计算所有点光源
		for(int i = 0; i < min(u_PointLightCount, MAX_POINT_LIGHT); i++)
		{
			color.rgb += CalcPointLight(u_PointLights[i]);
		}
        //计算聚光灯
		for(int i = 0; i < u_SpotLightCount; ++i)
		{
			color.rgb += CalcSpotLight(u_SpotLight);
		}	

		//自发光
		vec3 emission = texture(u_Material.emission, v_TexCoord).rgb;
		color.rgb += emission;
	
		//将计算得到的颜色值限制在[0.0, 1.0]的范围内->避免颜色值超出范围导致的渲染错误
		color = min(color, vec4(1.0));

		if(PostPorcess == 1)
		{
            //反相后处理
			color.rgb = vec3(1.0) - color.rgb;
		}
		if(PostPorcess == 2)
		{
            //加权灰度图后处理
			float average = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
			color = vec4(average, average, average, 1.0);
		}
	}
```

# 基于物理的光照模型
## 顶点着色器
```C++
//PBR Texture Shader

#type vertex
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    out vec2 TexCoords;
    out vec3 WorldPos;
    out vec3 Normal;

    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;

    void main()
    {
        TexCoords = aTexCoords;
        WorldPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(model) * aNormal;   
        gl_Position =  projection * view * vec4(WorldPos, 1.0);
    }

```
## 片段着色器
```C++

#type fragment
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoords;
    in vec3 WorldPos;
    in vec3 Normal;

    // 材质参数
    uniform sampler2D albedoMap;
    uniform sampler2D normalMap;
    uniform sampler2D metallicMap;
    uniform sampler2D roughnessMap;
    uniform sampler2D aoMap;

    // 光照参数
    uniform vec3 lightPositions[4];
    uniform vec3 lightColors[4];

    uniform vec3 camPos;

    const float PI = 3.14159265359;
//从法线贴图中获取物体表面的法向量
    vec3 getNormalFromMap()
    {
        //这里的2.0和1.0是为了将纹素颜色的范围从[0,1]映射到[-1,1]之间
        vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;
        //计算出纹理坐标和世界空间位置的偏导数
        vec3 Q1  = dFdx(WorldPos);
        vec3 Q2  = dFdy(WorldPos);
        vec2 st1 = dFdx(TexCoords);
        vec2 st2 = dFdy(TexCoords);

        vec3 N   = normalize(Normal);
        //计算切线(T)和副切线(B)向量
        //st2.t表示纹理坐标在t方向上的偏导数
        vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
        vec3 B  = -normalize(cross(N, T));
        //与物体表面的法向量(N)一起构成一个切空间(TBN)
        mat3 TBN = mat3(T, B, N);
        //将纹理空间中的切向量(tangentNormal)转换为世界空间中的法向量
        return normalize(TBN * tangentNormal);
    }
//微表面模型分布函数
//计算微表面的粗糙程度对光的散射影响，参数N表示物体表面的法向量，H表示半角向量，roughness表示粗糙度
    float DistributionGGX(vec3 N, vec3 H, float roughness)
    {
        //预先计算粗糙度和点积值的平方值
        float a = roughness*roughness;
        float a2 = a*a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH*NdotH;

        //分子
        float nom   = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        //分母
        denom = PI * denom * denom;
        //返回GGX分布函数的值
        return nom / denom;
    }

//计算GGX分布函数中的几何遮蔽因子，参数NdotV表示法向量N和视线方向V之间的点积值
    float GeometrySchlickGGX(float NdotV, float roughness)
    {
        float r = (roughness + 1.0);
        float k = (r*r) / 8.0;

        float nom   = NdotV;
        float denom = NdotV * (1.0 - k) + k;

        return nom / denom;
    }

//计算基于GGX分布函数的双向几何遮蔽因子
//参数N表示物体表面的法向量，V表示视线方向，L表示光线方向，roughness表示物体表面的粗糙度。
    float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
    {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2 = GeometrySchlickGGX(NdotV, roughness);
        float ggx1 = GeometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }

//于计算基于Schlick近似的菲涅尔反射系数
    vec3 fresnelSchlick(float cosTheta, vec3 F0)
    {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }

    void main()
    {		
        vec3 albedo     = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));
        float metallic  = texture(metallicMap, TexCoords).r;
        float roughness = texture(roughnessMap, TexCoords).r;
        float ao        = texture(aoMap, TexCoords).r;
        //获取物体表面的法向量N和视线方向V
        vec3 N = getNormalFromMap();
        vec3 V = normalize(camPos - WorldPos);

        //计算物体表面的菲涅尔反射系数F0
        vec3 F0 = vec3(0.04); 
        F0 = mix(F0, albedo, metallic);

        // 反射方程
        vec3 Lo = vec3(0.0);
        for(int i = 0; i < 4; ++i) 
        {
            // 计算每个光源的辐射能量
            //计算指向光源的光线方向向量L
            vec3 L = normalize(lightPositions[i] - WorldPos);
            // 视线方向V和光线方向L，计算反射向量H（半角向量）
            vec3 H = normalize(V + L);
            //计算出光源和物体表面点之间的距离distance
            float distance = length(lightPositions[i] - WorldPos);
            //计算光线的衰减因子attenuation，模拟光线在空气中传播时的衰减效果
            float attenuation = 1.0 / (distance * distance);
            vec3 radiance = lightColors[i] * attenuation;

            // 计算微表面反射模型中的双向反射分布函数
            //NDF表示法向量分布函数，G表示几何函数，F表示菲涅尔反射系数。
            float NDF = DistributionGGX(N, H, roughness);   
            float G   = GeometrySmith(N, V, L, roughness);      
            vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
           
            vec3 numerator    = NDF * G * F; 
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; 
            // + 0.0001是为了避免分母为0的情况
            //计算镜面反射分量
            vec3 specular = numerator / denominator;
        
            //将之前计算得到的菲涅尔反射系数F作为镜面反射系数kS
            vec3 kS = F;
            //计算漫反射部分的颜色系数
            vec3 kD = vec3(1.0) - kS;
            //计算结合金属度的漫反射颜色
            kD *= 1.0 - metallic;

            //NdotL表示光线方向L和物体表面法向量N的夹角的余弦值，它越大，则漫反射的贡献越大 
            float NdotL = max(dot(N, L), 0.0);        

            //(kD * albedo / PI + specular)->物体表面每个光源的总反射强度
            //将总反射强度乘以每个光源的辐射能量radiance，再乘以NdotL
            //得到物体表面每个光源的贡献值将这个贡献累加到最终颜色Lo上，得到物体表面的最终颜色。
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;  
        }   
        //计算环境光照，这里使用常量值0.03作为环境光系数，乘以反射率albedo和环境光遮蔽ao得到环境光照的颜色
        vec3 ambient = vec3(0.03) * albedo * ao;
    
        vec3 color = ambient + Lo;
        //将颜色color进行gamma校正，将rgb颜色转化为屏幕显示的颜色
        color = color / (color + vec3(1.0));
        color = pow(color, vec3(1.0/2.2)); 

        FragColor = vec4(color, 1.0);
    }
```
PBR（Physically-Based Rendering）模型公式，可以用于计算每个像素的颜色：

### 以下是PBR数学公式
片段着色器：

获取表面法向量：

$$\begin{aligned}\text{tangentNormal} &= \text{texture(normalMap, TexCoords).xyz} \times 2.0 - 1.0 \\ Q1 &= \text{dFdx(WorldPos)} \\ Q2 &= \text{dFdy(WorldPos)} \\ st1 &= \text{dFdx(TexCoords)} \\ st2 &= \text{dFdy(TexCoords)} \\ N &= \text{normalize(Normal)} \\ T &= \text{normalize(Q1}\times st2.t - Q2\times st1.t) \\ B &= -\text{normalize(cross(N, T))} \\ TBN &= \begin{bmatrix} T & B & N \end{bmatrix} \\ \text{getNormalFromMap} &= \text{normalize(TBN} \times \text{tangentNormal})\end{aligned}$$

微表面模型分布函数：

$$\begin{aligned} D_{GGX}(N, H, \text{roughness}) &= \frac{\text{a}^2}{\pi((N \cdot H)^2(\text{a}^2 - 1) + 1)^2} \\ a &= \text{roughness}^2 \end{aligned}$$

几何遮蔽因子：

$$\begin{aligned} G_{SchlickGGX}(N \cdot V, \text{roughness}) &= \frac{N \cdot V}{N \cdot V(1 - \frac{\text{roughness}+1}{2})+\frac{\text{roughness}+1}{2}} \end{aligned}$$

基于GGX分布函数的双向几何遮蔽因子：

$$\begin{aligned} G_{Smith}(N, V, L, \text{roughness}) &= G_{SchlickGGX}(N \cdot V, \text{roughness}) \times G_{SchlickGGX}(N \cdot L, \text{roughness}) \end{aligned}$$

菲涅尔反射系数：

$$\begin{aligned} F_{Schlick}(cos\theta, \text{F0}) &= \text{F0} + (1 - \text{F0}) \times (1 - \max(cos\theta, 0))^5 \end{aligned}$$

计算每个光源的镜面反射：

$$\begin{aligned} \text{numerator} &= D_{GGX}(N, H, \text{roughness}) \times G_{Smith}(N, V, L, \text{roughness}) \times F_{Schlick}(\max(N \cdot H, 0), \text{F0}) \\ \text{denominator} &= 4 \times \max(N \cdot V, 0) \times \max(N \cdot L, 0) + 0.0001 \\ \text{specular} &= \frac{\text{numerator}}{\text{denominator}} \end{aligned}$$

最终颜色：

$$\begin{aligned} \text{Lo} &= \sum_{i=0}^{3}(\frac{\text{kD} \times albedo}{\pi} + \text{specular}) \times \text{radiance} \times \max(N \cdot L, 0) \\ \text{ambient} &= 0.03 \times albedo \times ao \\ \text{color} &= \text{ambient} + \text{Lo} \\ \text{color} &= \frac{\text{color}}{\text{color} + 1} \\ \text{color} &= \text{pow(color, vec3(1.0/2.2))} \\ \text{FragColor} &= \begin{bmatrix} \text{color} \\ 1.0 \end{bmatrix} \end{aligned}$$


[def]: https://github.com/lqj126/Image/blob/main/%E6%AF%95%E4%B8%9A%E8%AE%BA%E6%96%87%E5%9B%BE%E7%89%87/%E7%94%9F%E6%88%90%E7%9A%84%E5%9B%BE%E7%89%87/7.png