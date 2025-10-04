import { Flex, Divider, Box, Image } from "@invoke-ai/ui-library";
import { memo, useCallback, useMemo, useRef, useState } from "react";
import { ImageComparisonDroppable } from "./ImageComparisonDroppable";
import { useImageDTO } from "services/api/endpoints/images";
import { useAppSelector } from "app/store/storeHooks";

import { selectImageToTokenize } from 'features/gallery/store/gallerySelectors';
import { useDebouncedMetadata } from "services/api/hooks/useDebouncedMetadata";
import { Dimensions } from "@xyflow/react";
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useStore } from '@nanostores/react';
import { fitDimsToContainer } from "./common";
import { TokenizationToolbar } from "./TokenizationToolbar";


export const ImageTokenization = memo(() => {
  const crossOrigin = useStore($crossOrigin);

  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  const imageDTO = useImageDTO(useAppSelector(selectImageToTokenize));
  const { metadata, isLoading } = useDebouncedMetadata(imageDTO?.image_name);

  // Ref callback runs synchronously when the DOM node is attached, ensuring we have a measurement before
  // the comparison content is rendered.
  const measureNode = useCallback((node: HTMLDivElement) => {
    if (node) {
      ref.current = node;
      const boundingRect = node.getBoundingClientRect();
      setRect(boundingRect);
    }
  }, []);

  const fittedDims = useMemo<Dimensions>(() => {
    if (!rect || !imageDTO) {
      return { width: 0, height: 0 };
    }
    return fitDimsToContainer(rect, imageDTO);
  }, [imageDTO, rect]);

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <TokenizationToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        <Box ref={measureNode} w="full" h="full" overflow="hidden">
          {imageDTO && 
            <Image  
              id="tokenization-image"
              src={imageDTO.image_url}
              fallbackSrc={imageDTO.thumbnail_url}
              crossOrigin={crossOrigin}
              w={fittedDims.width}
              h={fittedDims.height}
              maxW="full"
              maxH="full"
              objectFit="cover"
              objectPosition="top left"
            />
          }
          <pre>{JSON.stringify(metadata)}</pre>
        </Box>
        <ImageComparisonDroppable />
      </Flex>
    </Flex>
  );


});

ImageTokenization.displayName = 'ImageTokenization';
