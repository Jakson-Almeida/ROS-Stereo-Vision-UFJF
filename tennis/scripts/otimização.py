from cvxopt.modeling import op, variable

milho = variable(1, "Área de milho")
trigo = variable(1, "Área de trigo")
s = variable(4, "Folga")

# Definindo fob

fob = 600*milho[0] + 800*trigo[0]

# Definindo as restrições

rst = []

rst.append(milho[0] + trigo[0] + s[0] == 100)
rst.append(3*milho[0] + 2*trigo[0] + s[1] == 240)
rst.append(milho[0] + s[2] == 60)
rst.append(trigo[0] + s[3] == 80)
rst.append(milho[0] >= 0)
rst.append(trigo[0] >= 0)

for i in range(4):
    rst.append(s[i] >= 0)

# Montando o problema de otimização

agricultor = op(-fob, rst)

agricultor.solve("dense", "glpk")

print(f"O lucro é de R$ {fob.value()[0]:.2f}")
print("Sujeito à:")
print(f"{milho.name} é de {milho[0].value()[0]}")
print(f"{trigo.name} é de {trigo[0].value()[0]}")

##################################

PG1 = variable(1, "Geração 1")
PG2 = variable(1, "Geração 2")

# Definindo fob

fob = 10*PG1[0] + 20*PG2[0]

# Definindo as restrições

rst = []

rst.append(PG1[0] + PG2[0] == 10)
rst.append(PG1 <= 5)
rst.append(PG1 >= 0)
rst.append(PG2 <= 7)
rst.append(PG2 >= 1)

# Montando o problema de otimização

res = op(fob, rst)

res.solve("dense", "glpk")

print(f"O custo é de R$ {fob.value()[0]:.2f}")
print("Sujeito à:")
print(f"{PG1.name} é de {PG1[0].value()[0]}")
print(f"{PG2.name} é de {PG2[0].value()[0]}")

##############################################

from cvxopt.modeling import op, variable

PG1 = variable(1, "Geração 1")
PG2 = variable(1, "Geração 2")
s = variable(3, "Folga")
a = variable(2, "Folga")

# Definindo fob

fob = -3990*PG1[0] - 7980*PG2[0] + 4000*s[1] + 44000

# Definindo as restrições

rst = []

rst.append(PG1[0] + PG2[0] + a[0] == 10)
rst.append(PG1[0] + s[0] == 5)
rst.append(PG2[0] - s[1] + a[1] == 1)
rst.append(PG2[0] + s[2] == 7)
rst.append(PG1[0] >= 0)
rst.append(PG2[0] >= 0)
rst.append(s[0] >= 0)
rst.append(s[1] >= 0)
rst.append(s[2] >= 0)
rst.append(a[0] >= 0)
rst.append(a[1] >= 0)


# Montando o problema de otimização

energia = op(fob, rst)

energia.solve("dense", "glpk")

print(f"O custo é de R$ {fob.value()[0]:.2f}")
print("Sujeito à:")
print(f"{PG1.name} é de {PG1[0].value()[0]}")
print(f"{PG2.name} é de {PG2[0].value()[0]}")