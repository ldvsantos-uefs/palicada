--[[
Filtro Pandoc para colocar legendas de figuras ACIMA da imagem.
Por padrão, Pandoc coloca a legenda embaixo.
Este filtro inverte a ordem para: legenda primeiro, depois imagem.
]]

local TARGET_WIDTH = os.getenv("FIG_WIDTH") or "80%"

local function parse_length(s)
  if type(s) ~= "string" then
    return nil
  end
  local num, unit = s:match("^%s*([0-9]+%.?[0-9]*)%s*([A-Za-z%%]+)%s*$")
  if not num or not unit then
    return nil
  end
  return tonumber(num), unit
end

function Image(img)
  -- Evita distorção por dupla restrição, mantendo a proporção
  if img.attr and img.attr.attributes then
    if img.attr.attributes["height"] ~= nil then
      img.attr.attributes["height"] = nil
    end

    local w = img.attr.attributes["width"]
    local n, u = parse_length(w)

    -- Normaliza apenas imagens claramente "full width" em polegadas ou centímetros
    -- mantendo imagens pequenas, mosaicos e esquemas sem mexer
    if w == "6.5in" or w == "6.50in" then
      img.attr.attributes["width"] = TARGET_WIDTH
    elseif n and u == "in" and n >= 6.0 then
      img.attr.attributes["width"] = TARGET_WIDTH
    elseif n and u == "cm" and n >= 15.0 then
      img.attr.attributes["width"] = TARGET_WIDTH
    end
  end
  return img
end

function Figure(fig)
  -- Criar lista de blocos para retornar
  local result = {}
  
  -- Adicionar a legenda como parágrafo acima
  -- fig.caption.long contém uma lista de blocos com a legenda
  if fig.caption and fig.caption.long and #fig.caption.long > 0 then
    for _, block in ipairs(fig.caption.long) do
      table.insert(result, block)
    end
  end
  
  -- Adicionar a imagem sem legenda
  -- Criar uma nova figura sem caption, preservando atributos
  local img_only = pandoc.Figure(fig.content, pandoc.Caption())
  table.insert(result, img_only)
  
  -- Retornar os blocos na ordem: legenda, depois imagem
  return result
end
